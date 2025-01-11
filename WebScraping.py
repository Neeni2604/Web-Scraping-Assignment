import requests 
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin
from tqdm import tqdm
import pdfplumber
import re
import pandas as pd
import gender_guesser.detector as gender
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scholarly import scholarly
import time

# ============================
# Setup and Configuration
# ============================

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Defining the base URL
base_url = 'https://science.osti.gov/early-career'

# Directories to save PDFs and extracted text files
pdf_dir = 'award_pdfs'
txt_dir = 'pdf_texts'
os.makedirs(pdf_dir, exist_ok=True)
os.makedirs(txt_dir, exist_ok=True)



# ============================
# Step 1: Download PDFs
# ============================

def get_pdf_links(base_url):
    """
    Fetches all PDF links from the given base URL.
    """
    response = requests.get(base_url)
    if response.status_code != 200:
        raise Exception(f"Failed to load page {base_url}")

    soup = BeautifulSoup(response.content, 'html.parser')

    pdf_links_with_years = []
    for link in soup.find_all('a', href=True):
        if link['href'].lower().endswith('.pdf'):
            pdf_url = urljoin(base_url, link['href'])
            year_match = re.search(r'Fiscal Year (\d{4})', link.text)
            if year_match:
                year = int(year_match.group(1))
                pdf_links_with_years.append((pdf_url, year))
    logging.info(f"Found {len(pdf_links_with_years)} PDF files.")
    return pdf_links_with_years


def download_pdf(url, save_path):
    """
    Downloads a PDF from the given URL to the specified save path.
    """
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            return True
        else:
            logging.warning(f"Failed to download {url} - Status Code: {response.status_code}")
            return False
    except Exception as e:
        logging.error(f"Error downloading {url}: {e}")
        return False


def download_pdfs(pdf_links, pdf_dir):
    """
    Downloads all PDFs from the list of PDF links.
    """
    for pdf_url, year in tqdm(pdf_links, desc='Downloading PDFs'):
        pdf_name = pdf_url.split('/')[-1]
        save_path = os.path.join(pdf_dir, pdf_name)
        if not os.path.exists(save_path):
            success = download_pdf(pdf_url, save_path)
            if not success:
                logging.warning(f"Failed to download: {pdf_url}")
        else:
            logging.info(f"Already downloaded: {pdf_name}")



# ============================
# Step 2: Extract Text from PDFs
# ============================

def extract_text_to_txt(file_path, txt_save_path):
    """
    Extracts text from a PDF and saves it to a .txt file.
    """
    try:
        with pdfplumber.open(file_path) as pdf:
            text = ''
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + '\n'
        # Write the extracted text to the .txt file
        with open(txt_save_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(text)
        return True
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return False

def extract_texts(pdf_dir, txt_dir):
    """
    Processes all PDFs in the pdf_dir and extracts text to txt_dir.
    """
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]

    for pdf_file in tqdm(pdf_files, desc='Processing PDFs'):
        file_path = os.path.join(pdf_dir, pdf_file)
        txt_save_path = os.path.join(txt_dir, pdf_file.replace('.pdf', '.txt'))
        
        if not os.path.exists(txt_save_path):  # Avoid re-writing if already exists
            success = extract_text_to_txt(file_path, txt_save_path)
            if not success:
                logging.warning(f"Failed to extract text from: {pdf_file}")
        else:
            logging.info(f"Text already extracted for: {pdf_file}")

    logging.info(f"Text extraction completed. Text files saved to '{txt_dir}'.")



# ============================
# Step 3: Extract Names and Institutions
# ============================

# Initialize gender detector
detector = gender.Detector(case_sensitive=False)


# Define lists for categorization (extend these lists as needed)
national_labs = [
    'Argonne National Laboratory',
    'Brookhaven National Laboratory',
    'Fermilab',
    'Fermi National Accelerator Laboratory',
    'Lawrence Berkeley National Laboratory',
    'Los Alamos National Laboratory',
    'Oak Ridge National Laboratory',
    'Sandia National Laboratories',
    'National Renewable Energy Laboratory',
    'Pacific Northwest National Laboratory',
    'Lawrence Livermore National Laboratory',
    'Fermi National Accelerator Laboratory',
    'National Energy Technology Laboratory',
    'SLAC National Accelerator Laboratory',
    'Ames National Laboratory',
    'Thomas Jefferson National Accelerator Facility',
    'Princeton Plasma Physics Laboratory',
    'Idaho National Laboratory',
    'National Energy Technology Laboratory',
    'National Renewable Energy Laboratory',
    'Savannah River National Laboratory',
]

ivy_league = [
    'Harvard University',
    'Yale University',
    'Princeton University',
    'Columbia University',
    'University of Pennsylvania',
    'Dartmouth College',
    'Brown University',
    'Cornell University',
]

state_universities = [
    'University of California, Berkeley',
    'University of Michigan',
    'University of Texas at Austin',
    'University of Florida',
    'University of Wisconsin-Madison',
    'University of Illinois at Urbana-Champaign',
    'University of Washington',
    'Ohio State University',
    'University of North Carolina at Chapel Hill',
    'University of California, Los Angeles',
    'University of California, San Diego',
    'University of Maryland, College Park',
    'University of Minnesota',
    # Add more as needed
]

private_universities = [
    'Stanford University',
    'Massachusetts Institute of Technology',
    'California Institute of Technology',
    'Duke University',
    'Northwestern University',
    'Johns Hopkins University',
    'Rice University',
    'Emory University',
    'Washington University in St. Louis',
    'University of Chicago',
    'Columbia University',  # Note: Columbia is both Ivy League and private
    'Cornell University',   # Cornell is both Ivy League and private
    'Princeton University', # Similarly
    'University of Southern California',
    'New York University',
    # Add more as needed
]


def categorize_institution(institution):
    """
    Categorizes the institution into National Lab, Ivy League, State/Public, Private, or Other.
    """
    institution_lower = institution.lower()
    for nl in national_labs:
        if nl.lower() in institution_lower:
            return ('National Lab', None)
    for iv in ivy_league:
        if iv.lower() in institution_lower:
            return ('University', 'Ivy League')
    for st in state_universities:
        if st.lower() in institution_lower:
            return ('University', 'State/Public')
    for pr in private_universities:
        if pr.lower() in institution_lower:
            return ('University', 'Private')
    # If "University" is in the name but not in any list
    if 'university' in institution_lower:
        return ('University', 'Other')
    # If "College" is in the name but not in any list
    if 'college' in institution_lower:
        return ('University', 'Other')
    # Default category
    return ('Unknown', None)


def classify_gender(name):
    """
    Classifies gender based on the first name using gender_guesser.
    """
    first_name = name.split()[0]
    gender_prediction = detector.get_gender(first_name)
    if gender_prediction in ['male', 'mostly_male']:
        return 'Male'
    elif gender_prediction in ['female', 'mostly_female']:
        return 'Female'
    else:
        return 'Unknown'


def extract_entries_from_text(text):
    """
    Extracts (Name, Institution, Zipcode, Program Area) from the provided text.
    """
    lines = text.split('\n')
    entries = []
    department_keywords = ['department', 'division', 'group', 'office', 'section', 'center']
    
    # Program areas to look for
    program_areas = [
        'Accelerator R&D and Production',
        'Advanced Scientific Computing Research',
        'Basic Energy Sciences',
        'Biological and Environmental Research',
        'Fusion Energy Sciences',
        'High Energy Physics',
        'Isotope R&D and Production',
        'Nuclear Physics'
    ]

    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('Dr. '):
            name_match = re.match(r'Dr\.\s+([A-Za-z.\-\' ]+?),', line)
            if name_match:
                name = name_match.group(1).strip()
                institution = None
                zipcode = None
                program_area = None

                # Look for institution and zipcode
                for j in range(i+1, len(lines)):
                    current_line = lines[j].strip()
                    
                    # Look for ZIP code
                    zip_match = re.search(r'\b\d{5}\b', current_line)
                    if zip_match and not zipcode:
                        zipcode = zip_match.group(0)
                        # Institution is the line before this
                        if j-1 > i:
                            potential_institution = lines[j-1].strip()
                            if not any(keyword in potential_institution.lower() for keyword in department_keywords):
                                institution = potential_institution
                            else:
                                if j-2 > i:
                                    potential_institution = lines[j-2].strip()
                                    if not any(keyword in potential_institution.lower() for keyword in department_keywords):
                                        institution = potential_institution

                # Look for program area
                for j in range(max(0, i-20), min(len(lines), i+20)):  # Search 20 lines before and after
                    current_line = lines[j].strip()
                    if "This research was selected for funding by" in current_line:
                        for area in program_areas:
                            if area in current_line:
                                program_area = area
                                break
                        if not program_area:  # If area not found in standard form, extract from the line
                            office_match = re.search(r'Office of (.*?)\.', current_line)
                            if office_match:
                                program_area = office_match.group(1).strip()

                if institution and zipcode:  # Only add entry if we found both institution and zipcode
                    # entries.append({
                    #     'Name': name,
                    #     'Institution': institution,
                    #     'Zipcode': zipcode,
                    #     'Program_Area': program_area
                    # })
                    entries.append((name, institution, zipcode, program_area))

    return entries


def extract_and_classify_data(txt_dir, pdf_links):
    """
    Extracts names, institutions, zipcodes, and program areas from all text files and classifies them.
    """
    all_entries = []
    txt_files = [os.path.join(txt_dir, f) for f in os.listdir(txt_dir) if f.lower().endswith('.txt')]
    year_map = {pdf_url.split('/')[-1]: year for pdf_url, year in pdf_links}

    for txt_file in tqdm(txt_files, desc='Extracting Data from Texts'):
        try:
            with open(txt_file, 'r', encoding='utf-8') as file:
                text = file.read()
            
            entries = extract_entries_from_text(text)
            pdf_name = os.path.basename(txt_file).replace('.txt', '.pdf')
            year = year_map.get(pdf_name)
            
            if not entries:
                logging.warning(f"No entries found in {os.path.basename(txt_file)}")
                continue

            # for entry in entries:
            #     name = entry['Name']
            #     gender_class = classify_gender(name)
            #     category, subcategory = categorize_institution(entry['Institution'])
                
            #     entry.update({
            #         'Gender': gender_class,
            #         'Category': category,
            #         'Subcategory': subcategory
            #     })
            #     all_entries.append(entry)

            # With h-index
            for name, institution, zipcode, program_area in entries:
                gender_class = classify_gender(name)
                category, subcategory = categorize_institution(institution)
                
                # Look up h-index on Google Scholar
                h_index = get_h_index(name)
                
                all_entries.append({
                    'Name': name,
                    'Gender': gender_class,
                    'h-index': h_index,  # Add h-index to the entry
                    'Institution': institution,
                    'Category': category,
                    'Subcategory': subcategory,
                    'Zipcode' : zipcode,
                    'Program Area' : program_area,
                    'Year': year
                })
                time.sleep(5)  # Sleep to avoid too many requests too quickly
        
        except Exception as e:
            logging.error(f"Error processing {txt_file}: {e}")

    # Create DataFrame
    df = pd.DataFrame(all_entries)
    return df




# ============================
# Step 4: Look up h-index on Google Scholar
# ============================

def get_h_index(name):
    """
    Look up the h-index of a recipient using Google Scholar.
    Uses the scholarly library to search for the recipient and return their h-index.
    """
    try:
        search_query = scholarly.search_author(name)  # Correctly use the search_author method
        author = next(search_query, None)  # Get the first matching result
        
        if author:
            author_profile = scholarly.fill(author)  # Fetch full profile information
            h_index = author_profile.get('hindex', None)  # Get the h-index, or None if not found
            if h_index is not None:
                return h_index
            else:
                print(f"No h-index found for {name}.")
                return 'N/A'
        else:
            print(f"No Google Scholar profile found for {name}.")
            return 'N/A'
    except Exception as e:
        logging.error(f"Error looking up h-index for {name}: {e}")
        return 'N/A'

def extract_and_classify_data_with_h_index(txt_dir):
    """
    Extracts names and institutions from all text files, classifies them, and looks up their h-index.
    """
    all_entries = []
    txt_files = [os.path.join(txt_dir, f) for f in os.listdir(txt_dir) if f.lower().endswith('.txt')]

    for txt_file in tqdm(txt_files, desc='Extracting Data from Texts'):
        try:
            with open(txt_file, 'r', encoding='utf-8') as file:
                text = file.read()
            
            entries = extract_entries_from_text(text)
            
            if not entries:
                logging.warning(f"No entries found in {os.path.basename(txt_file)}")
                continue

            for name, institution in entries:
                gender_class = classify_gender(name)
                category, subcategory = categorize_institution(institution)
                
                # Look up h-index on Google Scholar
                h_index = get_h_index(name)
                
                all_entries.append({
                    'Name': name,
                    'Gender': gender_class,
                    'Institution': institution,
                    'Category': category,
                    'Subcategory': subcategory,
                    'h-index': h_index  # Add h-index to the entry
                })
                time.sleep(5)  # Sleep to avoid too many requests too quickly
        
        except Exception as e:
            logging.error(f"Error processing {txt_file}: {e}")

    # Create DataFrame
    df = pd.DataFrame(all_entries)
    return df



# ============================
# Step 5: Main Execution 
# ============================

def main():
    # Step 1: Download PDFs
    pdf_links = get_pdf_links(base_url)
    download_pdfs(pdf_links, pdf_dir)

    # Step 2: Extract Text from PDFs
    extract_texts(pdf_dir, txt_dir)

    # Step 3: Extract Names and Institutions, and retrieve h-index
    # df = extract_and_classify_data_with_h_index(txt_dir)
    df = extract_and_classify_data(txt_dir, pdf_links)
    
    if df.empty:
        logging.warning("No data extracted. Exiting.")
        return

    # Display the first few entries
    print(df.head())

    # Step 4: Save to CSV
    output_csv = 'classified_award_data.csv'
    df.to_csv(output_csv, index=False)
    logging.info(f"Data extraction and classification completed. Saved to '{output_csv}'.")

if __name__ == "__main__":
    main()
