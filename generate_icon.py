"""Generate app icon (hammer) as PNG and ICO from code — no external SVG renderer needed."""
from PIL import Image, ImageDraw
import struct, io, os

SIZE = 256

def draw_icon(size=SIZE):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    s = size / 256  # scale factor

    # Handle (brown wood gradient effect)
    handle_pts = [
        (int(85*s), int(135*s)), (int(105*s), int(120*s)),
        (int(185*s), int(230*s)), (int(165*s), int(245*s)),
    ]
    d.polygon(handle_pts, fill=(170, 120, 60, 255))
    # Handle highlight
    hl_pts = [
        (int(90*s), int(133*s)), (int(105*s), int(122*s)),
        (int(180*s), int(228*s)), (int(170*s), int(240*s)),
    ]
    d.polygon(hl_pts, fill=(196, 149, 106, 255))

    # Hammer head (steel)
    head_pts = [
        (int(30*s), int(65*s)), (int(65*s), int(25*s)),
        (int(165*s), int(95*s)), (int(130*s), int(135*s)),
    ]
    d.polygon(head_pts, fill=(107, 123, 141, 255))
    # Head highlight
    hh_pts = [
        (int(35*s), int(68*s)), (int(65*s), int(30*s)),
        (int(85*s), int(43*s)), (int(55*s), int(81*s)),
    ]
    d.polygon(hh_pts, fill=(155, 170, 185, 255))
    # Head dark side
    ds_pts = [
        (int(120*s), int(105*s)), (int(165*s), int(95*s)),
        (int(130*s), int(135*s)),
    ]
    d.polygon(ds_pts, fill=(74, 85, 104, 255))

    # Neural network accent dots (cyan)
    cyan = (79, 195, 247, 255)
    for cx, cy, r in [(200, 45, 7), (225, 28, 5), (220, 65, 6), (245, 48, 4)]:
        cx, cy, r = int(cx*s), int(cy*s), int(r*s)
        d.ellipse((cx-r, cy-r, cx+r, cy+r), fill=cyan)
    # Neural network lines
    lines = [((200,45),(225,28)), ((200,45),(220,65)), ((225,28),(245,48)), ((220,65),(245,48))]
    for (x1,y1),(x2,y2) in lines:
        d.line((int(x1*s),int(y1*s),int(x2*s),int(y2*s)), fill=(79,195,247,150), width=max(1,int(2*s)))

    return img

def save_ico(img, path):
    """Save as .ico with multiple sizes."""
    sizes = [16, 32, 48, 64, 128, 256]
    imgs = []
    for sz in sizes:
        resized = img.resize((sz, sz), Image.LANCZOS)
        imgs.append(resized)
    imgs[0].save(path, format="ICO", sizes=[(sz, sz) for sz in sizes], append_images=imgs[1:])

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "assets")
    os.makedirs(out_dir, exist_ok=True)

    img = draw_icon(256)
    img.save(os.path.join(out_dir, "icon.png"), "PNG")
    save_ico(img, os.path.join(out_dir, "icon.ico"))

    # Also save a 48px version for toolbar
    img48 = draw_icon(48)
    img48.save(os.path.join(out_dir, "icon_48.png"), "PNG")

    print("Generated: assets/icon.png, assets/icon.ico, assets/icon_48.png")
