import json
import os
from geopy.geocoders import Nominatim
import time

# Configuration
CACHE_DIR = r"D:\Cursor AI projects\Capstone2.1"
METADATA_CACHE = os.path.join(CACHE_DIR, 'cat_metadata_cache.json')
FOUND_CATS_METADATA = os.path.join(CACHE_DIR, 'found_cats_metadata.json')

def update_coordinates():
    print("🚀 Starting coordinate update...")
    
    # Initialize geocoder
    geolocator = Nominatim(user_agent="pet_id_malaysia_updater")
    
    # Load metadata
    if os.path.exists(METADATA_CACHE):
        with open(METADATA_CACHE, 'r') as f:
            metadata = json.load(f)
    else:
        print("❌ Metadata cache not found")
        return

    updated_count = 0
    
    for cat_id, data in metadata.items():
        # Only process found cats that don't have coordinates
        if data.get('status') == 'found' and ('lat' not in data or not data['lat']):
            location = data.get('location', 'Unknown')
            if location and location != 'Unknown':
                print(f"📍 Geocoding {cat_id}: {location}...")
                try:
                    # Add Malaysia to context if not present
                    query = location if "Malaysia" in location else f"{location}, Malaysia"
                    loc = geolocator.geocode(query)
                    
                    if loc:
                        data['lat'] = loc.latitude
                        data['lng'] = loc.longitude
                        print(f"   ✅ Found: {loc.latitude}, {loc.longitude}")
                        updated_count += 1
                    else:
                        # Default to Kuala Lumpur if not found, just for visualization
                        print("   ⚠️ Location not found, defaulting to KL")
                        data['lat'] = 3.1390
                        data['lng'] = 101.6869
                        updated_count += 1
                        
                    # Sleep to respect rate limits
                    time.sleep(1)
                except Exception as e:
                    print(f"   ❌ Error: {e}")
    
    if updated_count > 0:
        # Save main metadata
        with open(METADATA_CACHE, 'w') as f:
            json.dump(metadata, f)
        print(f"💾 Updated {updated_count} cats in main cache")
        
        # Update found cats specific metadata file
        if os.path.exists(FOUND_CATS_METADATA):
            with open(FOUND_CATS_METADATA, 'r') as f:
                found_cats = json.load(f)
            
            for cat_id, data in found_cats.items():
                if cat_id in metadata:
                    data['lat'] = metadata[cat_id].get('lat')
                    data['lng'] = metadata[cat_id].get('lng')
            
            with open(FOUND_CATS_METADATA, 'w') as f:
                json.dump(found_cats, f, indent=2)
            print("💾 Updated found cats metadata file")
            
    print("✨ Done!")

if __name__ == "__main__":
    update_coordinates()
