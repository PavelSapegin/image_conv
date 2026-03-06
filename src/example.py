from PIL import Image, ImageFilter

img = Image.open("image.jpg").convert("RGB")

filters = [
    ImageFilter.BLUR,
    ImageFilter.SHARPEN,
    ImageFilter.EDGE_ENHANCE,
    ImageFilter.SMOOTH,
]

for k, f in enumerate(filters):
    result = img.filter(f)
    result.save(f"output{k}.jpg")
