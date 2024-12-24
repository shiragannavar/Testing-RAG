import base64
import io
import os
from pdf2image import convert_from_path
from PIL import Image, ImageDraw


def pdf_to_base64url(pdf_path):
    # Convert PDF to a list of PIL Image objects
    pages = convert_from_path(pdf_path)

    # If the PDF has fewer than 2 pages, no side-by-side images can be produced
    if len(pages) < 2:
        return []

    output_images = []

    # Iterate through pages in pairs
    for i in range(len(pages) - 1):
        img1 = pages[i]
        img2 = pages[i + 1]

        # Get dimensions of both images
        w1, h1 = img1.size
        w2, h2 = img2.size

        # The combined image width is sum of both widths
        # Height is the max height of the two pages
        combined_width = w1 + w2
        combined_height = max(h1, h2)

        # Create a new blank image with white background
        new_img = Image.new('RGB', (combined_width, combined_height), (255, 255, 255))

        # Paste the two pages side by side
        new_img.paste(img1, (0, 0))
        new_img.paste(img2, (w1, 0))

        # Draw a vertical line at the boundary between the two pages
        draw = ImageDraw.Draw(new_img)
        line_x = w1  # vertical position where the first image ends and second starts
        line_color = (0, 0, 0)  # black line
        line_width = 3
        draw.line((line_x, 0, line_x, combined_height), fill=line_color, width=line_width)

        # Convert the combined image to base64url
        buffer = io.BytesIO()
        new_img.save(buffer, format="PNG")
        buffer.seek(0)
        b64_data = base64.urlsafe_b64encode(buffer.read()).decode('utf-8')

        output_images.append(b64_data)

    return output_images

# Example usage:
# result = pdf_pages_to_side_by_side_images_base64url("System of Automation.pdf")
# print(result)  # list of base64url-encoded strings
