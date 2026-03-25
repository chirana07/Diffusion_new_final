
from PIL import Image

inp = Image.open("eval15/low/22.png")
out = Image.open("eval_results/eval_22.png")

print("input size :", inp.size)
print("output size:", out.size)