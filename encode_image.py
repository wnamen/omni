#!/usr/bin/env python3
"""
Simple script to encode an image file to base64 and print it to stdout.
Usage: python encode_image.py <image_path>
"""

import base64
import sys
from pathlib import Path


def encode_image_to_base64(image_path):
    """Convert image file to base64 string and print it."""
    try:
        # Check if file exists
        if not Path(image_path).exists():
            print(f"Error: Image file {image_path} not found", file=sys.stderr)
            sys.exit(1)
        
        # Read and encode the image
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            encoded_string = encoded_string.decode('utf-8')
        
        # Print the base64 string to stdout
        print(encoded_string)
        
    except Exception as e:
        print(f"Error encoding image: {str(e)}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python encode_image.py <image_path>", file=sys.stderr)
        print("Example: python encode_image.py ios.png", file=sys.stderr)
        sys.exit(1)
    
    image_path = sys.argv[1]
    encode_image_to_base64(image_path)


if __name__ == "__main__":
    main() 