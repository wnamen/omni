#!/usr/bin/env python3
"""
Test script for the OmniParser API
Tests the /api/parse endpoint with mobile.png
"""

import base64
import json
import requests
import time
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:2171"
IMAGE_PATH = "ios.png"


def encode_image_to_base64(image_path):
    """Convert image file to base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            encoded_string = encoded_string.decode('utf-8')
        print(f"✓ Successfully encoded {image_path} to base64")
        return encoded_string
    except FileNotFoundError:
        print(f"✗ Error: Image file {image_path} not found")
        return None
    except Exception as e:
        print(f"✗ Error encoding image: {str(e)}")
        return None


def test_health_endpoint():
    """Test the health check endpoint."""
    try:
        print("Testing health endpoint...")
        response = requests.get(f"{API_BASE_URL}/api/health", timeout=10)
        
        if response.status_code == 200:
            health_data = response.json()
            print("✓ Health check passed")
            print(f"  Status: {health_data.get('status')}")
            print(f"  Models loaded: {health_data.get('models_loaded')}")
            print(f"  Device: {health_data.get('device')}")
            return True
        else:
            status = response.status_code
            print(f"✗ Health check failed with status {status}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"✗ Error: Cannot connect to {API_BASE_URL}")
        print("  Make sure the OmniParser service is running")
        return False
    except Exception as e:
        print(f"✗ Error testing health endpoint: {str(e)}")
        return False


def test_parse_endpoint(base64_image):
    """Test the parse endpoint with the provided image."""
    try:
        print("\nTesting parse endpoint...")
        print("⚠️  Note: This may take several minutes for large images...")
        
        # Prepare the request payload
        payload = {
            "image": base64_image
        }
        
        # Configure session with longer timeouts
        session = requests.Session()
        session.mount('http://', requests.adapters.HTTPAdapter(
            max_retries=0,
            pool_connections=1,
            pool_maxsize=1
        ))
        
        # Make the request with much longer timeout
        start_time = time.time()
        try:
            response = session.post(
                f"{API_BASE_URL}/api/parse",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=(30, 900)  # (connect timeout, read timeout) - 15 minutes for processing
            )
        except requests.exceptions.ReadTimeout:
            print("✗ Request timed out after 15 minutes")
            print("  The image processing is taking too long.")
            print("  Try with a smaller image or increase server resources.")
            return False
        except requests.exceptions.ConnectionError as e:
            if "Remote end closed connection" in str(e):
                print("✗ Server closed connection during processing")
                print("  This usually means the server ran out of memory or crashed.")
                print("  Try with a smaller image or check server logs.")
            else:
                print(f"✗ Connection error: {str(e)}")
            return False
            
        request_time = time.time() - start_time
        
        if response.status_code == 200:
            print(f"✓ Parse request successful (took {request_time:.2f}s)")
            
            # Parse response
            result = response.json()
            
            # Display results summary
            print("\nResults Summary:")
            print(f"  Elements found: {len(result.get('elements', []))}")
            
            # Display metadata about image processing
            if 'metadata' in result:
                metadata = result['metadata']
                original_size = metadata.get('original_size')
                processed_size = metadata.get('processed_size')
                was_resized = metadata.get('was_resized', False)
                
                print(f"  Original image size: {original_size}")
                print(f"  Processed image size: {processed_size}")
                if was_resized:
                    print("  ⚠️  Image was resized to prevent memory issues")
            
            if 'timing' in result:
                timing = result['timing']
                print(f"  OCR time: {timing.get('ocr_time', 0):.2f}s")
                print(f"  Parsing time: {timing.get('parsing_time', 0):.2f}s")
                total_time = timing.get('total_time', 0)
                print(f"  Total processing time: {total_time:.2f}s")
            
            # Display elements details
            elements = result.get('elements', [])
            if elements:
                print("\nDetected Elements:")
                # Show first 10 elements
                for i, element in enumerate(elements[:10]):
                    print(f"  {i+1}. Type: {element.get('type', 'N/A')}")
                    print(f"     Text: {element.get('text', 'N/A')}")
                    coords = element.get('coordinates', 'N/A')
                    print(f"     Coordinates: {coords}")
                    print()
                
                if len(elements) > 10:
                    remaining = len(elements) - 10
                    print(f"  ... and {remaining} more elements")
            
            # Save detailed results to file
            with open("parse_results.json", "w") as f:
                json.dump(result, f, indent=2)
            print("\n✓ Detailed results saved to parse_results.json")
            
            # Save labeled image if available
            if 'labeled_image' in result:
                try:
                    labeled_data = result['labeled_image']
                    labeled_image_data = base64.b64decode(labeled_data)
                    with open("labeled_img.png", "wb") as f:
                        f.write(labeled_image_data)
                    print("✓ Labeled image saved to labeled_img.png")
                except Exception as e:
                    print(f"✗ Error saving labeled image: {str(e)}")
            
            return True
            
        else:
            status_code = response.status_code
            print(f"✗ Parse request failed with status {status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Error testing parse endpoint: {str(e)}")
        return False


def main():
    """Main test function."""
    print("OmniParser API Test Script")
    print("=" * 40)
    
    # Check if image file exists
    if not Path(IMAGE_PATH).exists():
        print(f"✗ Error: {IMAGE_PATH} not found in current directory")
        return
    
    # Check image size and warn if it's large
    image_size = Path(IMAGE_PATH).stat().st_size / (1024 * 1024)  # Size in MB
    print(f"📁 Image size: {image_size:.1f} MB")
    if image_size > 5:
        print("⚠️  Large image detected - processing may take several minutes")
    
    # Test health endpoint first
    if not test_health_endpoint():
        return
    
    # Encode image to base64
    base64_image = encode_image_to_base64(IMAGE_PATH)
    if not base64_image:
        return
    
    # Test parse endpoint
    success = test_parse_endpoint(base64_image)
    
    if success:
        print("\n" + "=" * 40)
        print("✓ All tests completed successfully!")
        print("Check parse_results.json for detailed results")
        print("Check labeled_img.png for the processed image")
    else:
        print("\n" + "=" * 40)
        print("✗ Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()
