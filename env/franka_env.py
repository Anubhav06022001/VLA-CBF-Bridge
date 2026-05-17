from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT_DIR / "assets" / "franka_emika_panda"

SCENE_XML = ASSET_DIR / "scene.xml"

# self.model = mj.MjModel.from_xml_path(str(SCENE_XML))
