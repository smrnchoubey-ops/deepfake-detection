import requests
import os

# Create dummy image
with open('test_upload.jpg', 'wb') as f:
    f.write(os.urandom(1024)) # Just random bytes, might fail PIL, but let's try valid jpg header if needed.
    # Actually let's make a real small valid jpg to be safe
    from PIL import Image
    img = Image.new('RGB', (100, 100), color = 'red')
    img.save('test_upload.jpg')

url = 'http://127.0.0.1:10000/image-detect'
files = {'image': open('test_upload.jpg', 'rb')}

try:
    print(f"Sending POST request to {url}...")
    response = requests.post(url, files=files)
    print(f"Status Code: {response.status_code}")
    if "Error processing image" in response.text:
        print("FAIL: Server returned 'Error processing image'")
        # Extract error message if possible (simple split since we know the format now)
        try:
             # Look for "Error processing image: " and grab the rest of the line or tag
             # Simplified: just print the text around it
             idx = response.text.find("Error processing image")
             print("Error details excerpt:", response.text[idx:idx+200])
        except:
             pass
    else:
        print("SUCCESS: Image processed (or at least no generic error).")
        # print(response.text[:500]) # First 500 chars
except Exception as e:
    print(f"Request Failed: {e}")
