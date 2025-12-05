# Found Cats Feature - Documentation

## Overview

The Pet ID Malaysia website has been enhanced with a new "Found Cats" feature that allows users who have found missing cats to upload their photos to the database. This creates a two-way system:

1. **Lost Pet Owners** - Can upload their pet's photo to search the database for matches
2. **Found Cat Reporters** - Can upload photos of found cats to help reunite them with owners

---

## New Features Added

### 1. Found Cat Upload Page (`/found-cat`)
A dedicated page where users can upload photos of cats they've found, along with important details:
- Cat photo upload
- Location where the cat was found
- Date found
- Description of the cat
- Contact information (optional)

### 2. View Found Cats Gallery (`/view-found-cats`)
A gallery page displaying all cats that have been reported as found:
- Grid layout with cat photos
- Location and date information
- Contact information for each finder
- Ability to search through found cats

### 3. Success Confirmation Page
After uploading a found cat, users see a confirmation page showing:
- Upload success message
- Unique cat ID
- Preview of uploaded photo
- All submitted details

### 4. Enhanced Navigation
All pages now include links to:
- Home (Lost Pet Search)
- Found a Cat?
- View Found Cats
- About
- Contact

---

## Technical Implementation

### Backend Changes (app.py)

#### New Configuration Variables
```python
FOUND_CATS_PATH = r"D:\...\found_cats_dataset"  # Directory for found cat images
FOUND_CATS_METADATA = 'found_cats_metadata.json'  # Metadata storage
```

#### New Routes

1. **`@app.route('/found-cat')`**
   - Renders the found cat upload form

2. **`@app.route('/upload-found-cat', methods=['POST'])`**
   - Handles found cat photo uploads
   - Extracts AI embeddings from uploaded images
   - Saves metadata (location, date, description, contact)
   - Adds cat to the searchable database
   - Updates cache files

3. **`@app.route('/view-found-cats')`**
   - Displays all found cats from the database
   - Filters and sorts by upload time

4. **`@app.route('/found_cat_images/<path:filename>')`**
   - Serves found cat images from the found_cats_dataset directory

#### New Helper Function

**`save_found_cats_metadata()`**
- Saves found cat metadata to a separate JSON file
- Allows separate tracking of found cats vs. existing dataset

### Frontend Changes

#### New Templates Created

1. **`found_cat.html`**
   - Upload form with drag-and-drop functionality
   - Input fields for location, date, description, contact info
   - Real-time file validation
   - Loading overlay during upload

2. **`found_cat_success.html`**
   - Success confirmation page
   - Displays uploaded cat details
   - Shows preview of uploaded image
   - Links to upload more or view gallery

3. **`view_found_cats.html`**
   - Gallery view of all found cats
   - Card-based layout with hover effects
   - Contact modal for reaching finders
   - Empty state when no cats found

#### Updated Templates

All existing templates (`index.html`, `about.html`, `contact.html`, `results.html`) have been updated with:
- New navigation links to "Found a Cat?" and "View Found Cats"
- Consistent navigation across all pages

#### Homepage Enhancement

Added a new section to `index.html` with two clear options:
- **Lost Your Cat?** - Search for your pet
- **Found a Cat?** - Report a found cat

---

## How It Works

### For Users Who Found a Cat

1. Navigate to "Found a Cat?" in the menu
2. Upload a clear photo of the cat
3. Fill in required details:
   - Location where found (required)
   - Date found (auto-filled with today's date)
   - Description (optional but recommended)
   - Contact information (optional but helpful)
4. Submit the form
5. AI extracts features from the photo
6. Cat is added to the database
7. Success page confirms upload

### For Lost Pet Owners

1. Upload their pet's photo on the home page
2. AI searches through **entire database** including:
   - Original dataset cats
   - Newly uploaded found cats
3. Get matches with similarity scores
4. Contact information available if provided by finder

### AI Integration

- **Embedding Extraction**: Same EfficientNet-B3 model used for both lost and found cats
- **Real-time Database Updates**: Found cats immediately added to searchable database
- **Unified Search**: Lost pet search includes all cats (original dataset + found cats)
- **Metadata Storage**: Separate tracking for found cats while maintaining unified search

---

## Database Structure

### Found Cats Metadata Format
```json
{
  "found_20241114_123456": {
    "filename": "found_cat_20241114_123456_image.jpg",
    "image_path": "D:/...\\found_cats_dataset\\found_cat_20241114_123456_image.jpg",
    "id": "found_20241114_123456",
    "name": "Found Cat 20241114_123456",
    "location": "Near Central Park, Kuala Lumpur",
    "description": "Orange tabby with white paws...",
    "contact_info": "+60 12-345 6789",
    "date_found": "2024-11-14",
    "upload_time": "2024-11-14 12:34:56",
    "status": "found"
  }
}
```

### Storage Locations

1. **Images**: `found_cats_dataset/` directory
2. **Embeddings**: `cat_embeddings_cache.npy` (combined with original dataset)
3. **Metadata**: 
   - `cat_metadata_cache.json` (combined database)
   - `found_cats_metadata.json` (found cats only)

---

## User Interface Features

### Design Elements

- **Modern Bootstrap 5 UI**: Consistent with existing design
- **Responsive Layout**: Works on all devices
- **Drag-and-Drop Upload**: Easy file selection
- **Real-time Preview**: See image before upload
- **Loading Indicators**: Visual feedback during processing
- **Success Animations**: Clear confirmation of actions

### Color Scheme

- **Primary Blue**: Navigation and primary actions
- **Success Green**: Found cat actions and confirmations
- **Danger Red**: Lost pet search (urgent action)
- **Info Blue**: View gallery and information

### Icons (Bootstrap Icons)

- 🏠 Home
- 📷 Found a Cat
- 🔍 Search/Lost Pet
- 📊 View Gallery
- ✓ Success indicators

---

## Benefits of This Feature

### For the Community

1. **Two-Way System**: Both finders and owners can be proactive
2. **Increased Database**: More cats = better matching
3. **Community Engagement**: People help each other
4. **Real-time Updates**: No waiting for admin approval

### For Pet Owners

1. **More Chances**: Pets might be found before owner searches
2. **Contact Information**: Direct connection with finders
3. **Location Data**: Know where pet was seen
4. **Time Sensitivity**: Recent sightings are visible

### For Finders

1. **Easy Upload**: Simple, intuitive process
2. **Help Others**: Feel good about contributing
3. **No Account Needed**: Quick anonymous uploads possible
4. **Optional Contact**: Choose to be reached or not

---

## Security & Privacy

- **File Validation**: Only accepts JPG, JPEG, PNG formats
- **Size Limits**: 16MB maximum file size
- **Secure Storage**: Images stored in dedicated directory
- **Optional Contact**: Users choose whether to share contact info
- **No Personal Data Required**: Anonymous uploads possible

---

## Future Enhancements (Suggestions)

1. **Admin Moderation**: Review uploads before going live
2. **User Accounts**: Track uploads and matches
3. **Notifications**: Alert owners when similar cat is found
4. **Geographic Search**: Filter by location/distance
5. **Multiple Images**: Upload multiple photos per cat
6. **Status Updates**: Mark as reunited
7. **Social Sharing**: Share found cats on social media
8. **Mobile App**: Native mobile application

---

## Testing Checklist

- [x] Found cat upload form works
- [x] Image validation (type, size)
- [x] Drag and drop functionality
- [x] AI embedding extraction for found cats
- [x] Database update with new cats
- [x] Success page displays correctly
- [x] Gallery view shows all found cats
- [x] Contact modal works
- [x] Navigation links work on all pages
- [x] Responsive design on mobile
- [x] Found cats appear in lost pet search results

---

## Deployment Notes

1. Ensure `found_cats_dataset` directory exists and is writable
2. Check that all template files are in the `templates/` directory
3. Verify static files (CSS, JS) are accessible
4. Test file upload permissions
5. Monitor disk space for image storage
6. Backup metadata JSON files regularly

---

## Support & Maintenance

### Regular Tasks

1. **Monitor uploads**: Check quality and appropriateness
2. **Backup data**: Regular backups of images and metadata
3. **Clean old entries**: Archive reunited pets
4. **Update database**: Refresh embeddings cache periodically

### Troubleshooting

**Issue**: Upload fails
- Check file size (< 16MB)
- Verify file format (JPG, JPEG, PNG)
- Ensure found_cats_dataset directory exists and is writable

**Issue**: Cat not appearing in search results
- Verify embedding was successfully extracted
- Check that cat was added to cat_embeddings and cat_metadata
- Restart application to reload cache

**Issue**: Images not displaying
- Check file path in metadata
- Verify found_cat_images route is working
- Ensure images are in found_cats_dataset directory

---

## Conclusion

This feature transforms Pet ID Malaysia from a one-way search tool into a comprehensive community-driven platform for reuniting lost pets. By enabling both pet owners and finders to actively participate, we increase the chances of successful reunions and create a more engaged user base.

The implementation maintains the same high-quality AI matching while adding valuable community features and maintaining ease of use.

