from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r"C:\Users\tmark\Documents\Codex\2026-08-15\referenced-chatgpt-conversation-this-is-an")
DATA = ROOT / "work" / "gwtc5_joint_mechanism_exploratory" / "joint_geometry_event_metrics.csv"
OUTPUT = ROOT / "work" / "gwtc5_joint_mechanism_exploratory" / "figure_joint_mechanism.png"

W, H = 2400, 1050
BG = "white"
GRID = "#dfe6ee"
TEXT = "#18212b"
AXIS = "#344250"
CONF = "#256b9a"
REPL = "#d97706"
LIGHT_CONF = "#a7c7dd"
LIGHT_REPL = "#f3c58d"
FONT_PATH = Path(r"C:\Windows\Fonts\arial.ttf")
BOLD_PATH = Path(r"C:\Windows\Fonts\arialbd.ttf")


def font(size: int, bold: bool = False):
    return ImageFont.truetype(str(BOLD_PATH if bold else FONT_PATH), size)


def read_rows():
    with DATA.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in (
            "joint_q_chi_eff_sliced_w1",
            "chi_eff_nw1",
            "chi_p_nw1",
        ):
            row[key] = float(row[key])
    return rows


def triangle(draw, x, y, radius, fill, outline=None):
    pts = [(x, y-radius), (x-radius*0.9, y+radius*0.78), (x+radius*0.9, y+radius*0.78)]
    draw.polygon(pts, fill=fill, outline=outline or fill)


def circle(draw, x, y, radius, fill, outline=None):
    draw.ellipse((x-radius,y-radius,x+radius,y+radius), fill=fill, outline=outline or fill, width=2)


def main():
    rows = read_rows()
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.text((W//2, 34), "Exploratory joint mass-ratio-spin analysis", font=font(48, True), fill=TEXT, anchor="ma")

    panels = [(135, 155, 1120, 900), (1320, 155, 2270, 900)]
    for label, panel in zip(("A", "B"), panels):
        x0,y0,x1,y1 = panel
        draw.rounded_rectangle(panel, radius=20, fill="#fbfcfe", outline="#c7d2de", width=3)
        draw.text((x0+22,y0+18), label, font=font(38, True), fill=TEXT)

    # Panel A: joint sliced W1 versus chi_eff NW1.
    x0,y0,x1,y1 = panels[0]
    left, right, top, bottom = x0+125, x1-40, y0+92, y1-110
    x_max, y_max = 0.16, 0.115
    def sx(v): return left + v/x_max*(right-left)
    def sy(v): return bottom - v/y_max*(bottom-top)
    for tick in (0,0.04,0.08,0.12,0.16):
        x=sx(tick); draw.line((x,top,x,bottom),fill=GRID,width=2)
        draw.text((x,bottom+18),f"{tick:.2f}",font=font(25),fill=TEXT,anchor="ma")
    for tick in (0,0.025,0.05,0.075,0.10):
        y=sy(tick); draw.line((left,y,right,y),fill=GRID,width=2)
        draw.text((left-18,y),f"{tick:.3f}" if tick else "0",font=font(25),fill=TEXT,anchor="rm")
    draw.line((left,top,left,bottom),fill=AXIS,width=4); draw.line((left,bottom,right,bottom),fill=AXIS,width=4)
    for row in rows:
        x=sx(row["chi_eff_nw1"]); y=sy(row["joint_q_chi_eff_sliced_w1"])
        if row["stage"]=="confirmatory": circle(draw,x,y,10,CONF,"white")
        else: triangle(draw,x,y,12,REPL,"white")
    draw.text(((left+right)//2,bottom+65),"chi_eff NW1",font=font(31,True),fill=TEXT,anchor="ma")
    # PIL cannot rotate anchors cleanly through draw.text; render a rotated label.
    label_img=Image.new("RGBA",(620,55),(255,255,255,0)); ld=ImageDraw.Draw(label_img); ld.text((310,27),"Joint (q, chi_eff) sliced W1",font=font(31,True),fill=TEXT,anchor="mm")
    label_img=label_img.rotate(90,expand=True,resample=Image.Resampling.BICUBIC)
    image.paste(label_img,(x0+18,(top+bottom)//2-label_img.height//2),label_img)
    draw.rectangle((left+28,top+18,left+430,top+112),fill="#ffffffdd",outline="#c7d2de",width=2)
    draw.text((left+48,top+34),"Confirmatory: rho = 0.965",font=font(27,True),fill=CONF)
    draw.text((left+48,top+72),"Replication: rho = 0.900",font=font(27,True),fill=REPL)

    # Panel B: paired chi_p and chi_eff NW1.
    x0,y0,x1,y1 = panels[1]
    left, right, top, bottom = x0+115, x1-45, y0+130, y1-110
    y_max=0.16
    def py(v): return bottom - v/y_max*(bottom-top)
    xp, xe = left+210, right-210
    for tick in (0,0.04,0.08,0.12,0.16):
        y=py(tick); draw.line((left,y,right,y),fill=GRID,width=2)
        draw.text((left-18,y),f"{tick:.2f}",font=font(25),fill=TEXT,anchor="rm")
    draw.line((left,top,left,bottom),fill=AXIS,width=4); draw.line((left,bottom,right,bottom),fill=AXIS,width=4)
    for idx,row in enumerate(rows):
        jitter=((idx%7)-3)*2.5
        y_p=py(row["chi_p_nw1"]); y_e=py(row["chi_eff_nw1"])
        color=CONF if row["stage"]=="confirmatory" else REPL
        light=LIGHT_CONF if row["stage"]=="confirmatory" else LIGHT_REPL
        draw.line((xp+jitter,y_p,xe+jitter,y_e),fill=light,width=3)
        if row["stage"]=="confirmatory":
            circle(draw,xp+jitter,y_p,7,color,"white"); circle(draw,xe+jitter,y_e,7,color,"white")
        else:
            triangle(draw,xp+jitter,y_p,9,color,"white"); triangle(draw,xe+jitter,y_e,9,color,"white")
    draw.text((xp,bottom+55),"chi_p",font=font(32,True),fill=TEXT,anchor="ma")
    draw.text((xe,bottom+55),"chi_eff",font=font(32,True),fill=TEXT,anchor="ma")
    label_img2=Image.new("RGBA",(220,55),(255,255,255,0)); ld2=ImageDraw.Draw(label_img2); ld2.text((110,27),"NW1",font=font(31,True),fill=TEXT,anchor="mm")
    label_img2=label_img2.rotate(90,expand=True,resample=Image.Resampling.BICUBIC)
    image.paste(label_img2,(x0+4,(top+bottom)//2-label_img2.height//2),label_img2)
    draw.text(((left+right)//2,y0+34),"chi_eff exceeds chi_p in 18/18 and 9/9 events",font=font(28,True),fill=TEXT,anchor="ma")
    draw.text(((left+right)//2,y0+72),"Ordering also persists with KS distance",font=font(26),fill=TEXT,anchor="ma")

    # Legend and footnote.
    circle(draw,140,H-70,10,CONF,"white"); draw.text((165,H-70),"18-event confirmatory stage",font=font(27),fill=TEXT,anchor="lm")
    triangle(draw,600,H-70,12,REPL,"white"); draw.text((625,H-70),"9-event replication stage",font=font(27),fill=TEXT,anchor="lm")
    draw.text((W-75,H-70),"Post-confirmatory exploratory analysis",font=font(26),fill="#5b6773",anchor="rm")

    image.save(OUTPUT,quality=95,dpi=(300,300))
    print(OUTPUT)

if __name__ == "__main__":
    main()
