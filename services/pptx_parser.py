from pathlib import Path
from typing import List, Dict, Any
import zipfile
import xml.etree.ElementTree as ET

class PresentationParser:
    """
    Implements PPTX slide and shape hierarchy traversal, defensive text extraction,
    and deterministic slide-quality audit heuristics as specified in Section 4.10 and 5.9.
    
    Supports dual execution engines:
    - Primary: python-pptx (high-level object model)
    - Fallback: Native Python standard-library OpenXML zip parser (zero external dependencies)
    """
    def __init__(self):
        pass

    def parse(self, path: Path) -> List[Dict[str, Any]]:
        try:
            return self._parse_with_pptx(path)
        except ImportError:
            # Seamless fallback to built-in OpenXML XML parser if pptx module is not found
            return self._parse_with_openxml(path)
        except Exception:
            # If python-pptx fails for any reason, fallback to OpenXML
            return self._parse_with_openxml(path)

    def _parse_with_pptx(self, path: Path) -> List[Dict[str, Any]]:
        from pptx import Presentation

        prs = Presentation(str(path))
        output = []

        for slide_number, slide in enumerate(prs.slides, start=1):
            texts = []
            word_count = 0
            has_picture = False
            small_font_runs = 0

            for shape in slide.shapes:
                # 13 corresponds to MSO_SHAPE_TYPE.PICTURE
                if shape.shape_type == 13:
                    has_picture = True
                
                # Check for table, chart, or visual graphic shape
                if shape.has_table or shape.has_chart:
                    has_picture = True

                # Check text frame
                if not getattr(shape, "has_text_frame", False):
                    continue

                for paragraph in shape.text_frame.paragraphs:
                    for run in paragraph.runs:
                        value = run.text.strip()
                        if value:
                            texts.append(value)
                            word_count += len(value.split())

                        size = run.font.size
                        if size is not None and size.pt < 18:
                            small_font_runs += 1

            warnings = []

            # Rule 1: Text density heuristic (> 40 words)
            if word_count > 40:
                warnings.append("Text exceeds 40 words. Consider splitting the slide or moving detail to speaker notes.")

            # Rule 2: Visual balance heuristic (> 10 words, no picture)
            if word_count > 10 and not has_picture:
                warnings.append("Text-heavy slide has no detected picture. Consider adding a relevant chart, diagram or image.")

            # Rule 3: Typography heuristic (explicit font < 18 pt)
            if small_font_runs > 0:
                warnings.append(f"{small_font_runs} run(s) use explicit font size below 18 pt. Increase text size or reduce content density.")

            output.append({
                "slide": slide_number,
                "text": "\n".join(texts),
                "word_count": word_count,
                "has_picture": has_picture,
                "small_font_runs": small_font_runs,
                "warnings": warnings
            })

        return output

    def _parse_with_openxml(self, path: Path) -> List[Dict[str, Any]]:
        """
        Native zero-dependency OpenXML parser extracting slide text, pictures,
        and typography directly from the PPTX zip archive.
        """
        output = []
        with zipfile.ZipFile(str(path), "r") as z:
            # Find all slide XML files sorted numerically
            slide_files = [f for f in z.namelist() if f.startswith("ppt/slides/slide") and f.endswith(".xml")]
            
            # Sort by slide index (e.g. slide1, slide2, slide10)
            def slide_key(fname):
                num_part = "".join(c for c in Path(fname).stem if c.isdigit())
                return int(num_part) if num_part else 0
            slide_files.sort(key=slide_key)

            for slide_number, s_file in enumerate(slide_files, start=1):
                xml_data = z.read(s_file)
                tree = ET.fromstring(xml_data)

                # Check for pictures: <p:pic> or <a:blip>
                has_picture = any(
                    elem.tag.endswith("}pic") or elem.tag.endswith("}blip") or elem.tag.endswith("}graphic")
                    for elem in tree.iter()
                )

                # Extract text runs and check font sizes
                texts = []
                word_count = 0
                small_font_runs = 0

                for elem in tree.iter():
                    # Text elements: <a:t>
                    if elem.tag.endswith("}t") and elem.text:
                        val = elem.text.strip()
                        if val:
                            texts.append(val)
                            word_count += len(val.split())

                    # Text run properties: <a:rPr sz="1200"> (sz is 1/100 of a point, so 1200 = 12 pt)
                    if elem.tag.endswith("}rPr"):
                        sz = elem.attrib.get("sz")
                        if sz and sz.isdigit():
                            pt = int(sz) / 100.0
                            if pt < 18.0:
                                small_font_runs += 1

                warnings = []
                # Rule 1: Text density heuristic
                if word_count > 40:
                    warnings.append("Text exceeds 40 words. Consider splitting the slide or moving detail to speaker notes.")

                # Rule 2: Visual balance heuristic
                if word_count > 10 and not has_picture:
                    warnings.append("Text-heavy slide has no detected picture. Consider adding a relevant chart, diagram or image.")

                # Rule 3: Typography heuristic
                if small_font_runs > 0:
                    warnings.append(f"{small_font_runs} run(s) use explicit font size below 18 pt. Increase text size or reduce content density.")

                output.append({
                    "slide": slide_number,
                    "text": "\n".join(texts),
                    "word_count": word_count,
                    "has_picture": has_picture,
                    "small_font_runs": small_font_runs,
                    "warnings": warnings
                })

        return output
