from patches import PatchExtractor

extract_patches = PatchExtractor("data/shapefiles/fieldshape.geojson", "data/raw")
extract_patches.extract_all_orthos()

