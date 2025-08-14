import json
import re
import html
from pathlib import Path

INPUT_JSON = "data/sirius_raw_data.json"
OUTPUT_TXT = "data/sirius_raw_data.txt"

def html_to_text(s: str) -> str:
    if not s:
        return ""
    s = html.unescape(s)
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"<li[^>]*>", "- ", s, flags=re.I)
    s = re.sub(r"</(p|div|h\d|ul|ol)>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def split_tags(tags_field):
    """Return a list of tags from either a list or a comma/pipe/semicolon-separated string."""
    if not tags_field:
        return []
    if isinstance(tags_field, list):
        return [t.strip() for t in tags_field if str(t).strip()]
    # it's a string: split on commas / pipes / semicolons
    parts = re.split(r"[,\|;]", str(tags_field))
    return [p.strip() for p in parts if p.strip()]

def pick_price_and_stock(product):
    """Handle either variants[].price.current or min_price/max_price."""
    # Variant style
    variants = product.get("variants") or []
    if variants:
        v = variants[0] or {}
        price = ((v.get("price") or {}).get("current")) or 0
        stock = ((v.get("price") or {}).get("stockStatus")) or "Unknown"
        if price:
            return f"${price/100:.2f}", stock
        return "N/A", stock

    # Catalog style
    min_p = product.get("min_price")
    max_p = product.get("max_price")
    if isinstance(min_p, (int, float)) and min_p > 0:
        # prefer min price if available
        return f"${float(min_p):.2f}", product.get("stockStatus", "Unknown")
    if isinstance(max_p, (int, float)) and max_p > 0:
        return f"${float(max_p):.2f}", product.get("stockStatus", "Unknown")

    return "N/A", "Unknown"

def normalize_product(product: dict) -> str:
    # Title
    title = (product.get("title") or "").strip()

    # Brand/vendor
    brand = (product.get("brand") or product.get("vendor") or "").strip()

    # Categories
    cats_list = product.get("categories")
    if isinstance(cats_list, list) and cats_list:
        cats = "; ".join([str(c).strip() for c in cats_list if str(c).strip()])
    else:
        cats = (product.get("product_type") or "").strip()

    # Tags (handle list OR comma-separated string)
    tags_list = split_tags(product.get("tags"))
    tags = "; ".join(tags_list)

    # URL (either source.canonicalUrl OR url)
    url = (
        (product.get("source") or {}).get("canonicalUrl")
        or product.get("url")
        or ""
    ).strip()

    # Description / HTML
    desc_raw = product.get("description")
    if not desc_raw:
        desc_raw = product.get("html")  # alternate field in your raw example
    desc = html_to_text(desc_raw) if desc_raw else "(No description provided)"

    # Price + stock
    price, stock = pick_price_and_stock(product)

    block = [
        f"Title: {title}",
        f"Brand: {brand}",
        f"Categories: {cats}",
        f"Tags: {tags}",
        f"Price: {price}",
        f"URL: {url}",
        "",
        desc
    ]
    return "\n".join(block)

def main():
    data_path = Path(INPUT_JSON)
    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    with Path(OUTPUT_TXT).open("w", encoding="utf-8") as out:
        for product in data:
            text_block = normalize_product(product)
            out.write(text_block + "\n\n" + "="*80 + "\n\n")

    print(f"Normalized text written to {OUTPUT_TXT}")

if __name__ == "__main__":
    main()
