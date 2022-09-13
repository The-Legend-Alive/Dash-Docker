from bs4 import BeautifulSoup as bs4
from dash import html
from datetime import date, timedelta, datetime
from dotenv import load_dotenv, find_dotenv
from geopy.geocoders import GoogleV3
from imagekitio import ImageKit
from numpy import NaN
from os.path import exists
import logging
import os
import pandas as pd
import requests

## SETUP AND VARIABLES
load_dotenv(find_dotenv())
g = GoogleV3(api_key=os.getenv('GOOGLE_API_KEY')) # https://github.com/geopy/geopy/issues/171

logging.getLogger().setLevel(logging.INFO)

# ImageKit.IO
# https://docs.imagekit.io/api-reference/upload-file-api/server-side-file-upload#uploading-file-via-url
# Create keys at https://imagekit.io/dashboard/developer/api-keys
imagekit = ImageKit(
    public_key=os.getenv('IMAGEKIT_PUBLIC_KEY'),
    private_key=os.getenv('IMAGEKIT_PRIVATE_KEY'),
    url_endpoint = os.getenv('IMAGEKIT_URL_ENDPOINT')
)

# Make the dataframe a global variable
global df

### PANDAS DATAFRAME OPERATIONS
# Prompt user for the CSV filename
csv = input('Enter the CSV filename.\n')
# import the csv
# Don't round the float. See https://stackoverflow.com/a/68027847
# Convert all empty strings into NaNs. See https://stackoverflow.com/a/53075732
df = pd.read_csv(f"./{csv}", float_precision="round_trip", skipinitialspace=True)
pd.set_option("display.precision", 10)

# Strip leading and trailing whitespaces from the column names
# https://stackoverflow.com/a/36082588
df.columns = df.columns.str.strip()

# Drop all rows that don't have a city. # TODO: figure out a workaround
df = df[df['City'].notna()]

# Drop all rows that don't have a MLS Listing ID (MLS#) (aka misc data we don't care about)
# https://stackoverflow.com/a/13413845
df = df[df['Listing ID (MLS#)'].notna()]

# Create a new column with the Street Number & Street Name
df["Short Address"] = df["St#"] + ' ' + df["St Name"].str.strip() + ',' + ' ' + df['City']

# Define HTML code for the popup so it looks pretty and nice
def popup_html(row):
    i = row.Index
    street_address=df['Full Street Address'].at[i] 
    mls_number=df['Listing ID (MLS#)'].at[i]
    mls_number_hyperlink=df['bhhs_url'].at[i]
    mls_photo = df['MLS Photo'].at[i]
    lc_price = df['List Price'].at[i] 
    price_per_sqft=df['Price Per Square Foot'].at[i]                  
    brba = df['Br/Ba'].at[i]
    square_ft = df['Sqft'].at[i]
    year = df['YrBuilt'].at[i]
    garage = df['Garage Spaces'].at[i]
    pets = df['PetsAllowed'].at[i]
    phone = df['List Office Phone'].at[i]
    terms = df['Terms'].at[i]
    sub_type = df['Sub Type'].at[i]
    listed_date = pd.to_datetime(df['Listed Date'].at[i]).date() # Convert the full datetime into date only. See https://stackoverflow.com/a/47388569
    furnished = df['Furnished'].at[i]
    key_deposit = df['DepositKey'].at[i]
    other_deposit = df['DepositOther'].at[i]
    pet_deposit = df['DepositPets'].at[i]
    security_deposit = df['DepositSecurity'].at[i]
    # If there's no square footage, set it to "Unknown" to display for the user
    # https://towardsdatascience.com/5-methods-to-check-for-nan-values-in-in-python-3f21ddd17eed
    if pd.isna(square_ft) == True:
        square_ft = 'Unknown'
    # If there IS a square footage, convert it into an integer (round number)
    elif pd.isna(square_ft) == False:
        square_ft = f"{int(square_ft)} sq. ft"
    # Repeat above for Year Built
    if pd.isna(year) == True:
        year = 'Unknown'
    # If there IS a square footage, convert it into an integer (round number)
    elif pd.isna(year) == False:
        year = f"{int(year)}"
    # Repeat above for garage spaces
    if pd.isna(garage) == True:
        garage = 'Unknown'
    elif pd.isna(garage) == False:
        garage = f"{int(garage)}"
    # Repeat for ppsqft
    if pd.isna(price_per_sqft) == True:
        price_per_sqft = 'Unknown'
    elif pd.isna(price_per_sqft) == False:
        price_per_sqft = f"${float(price_per_sqft)}"
    # Repeat for listed date
    if pd.isna(listed_date) == True:
        listed_date = 'Unknown'
    elif pd.isna(listed_date) == False:
        listed_date = f"{listed_date}"
    # Repeat for furnished
    if pd.isna(furnished) == True:
        furnished = 'Unknown'
    elif pd.isna(furnished) == False:
        furnished = f"{furnished}"
    # Repeat for the deposits
    if pd.isna(key_deposit) == True:
        key_deposit = 'Unknown'
    elif pd.isna(key_deposit) == False:
        key_deposit = f"${int(key_deposit)}"
    if pd.isna(pet_deposit) == True:
        pet_deposit = 'Unknown'
    elif pd.isna(pet_deposit) == False:
        pet_deposit = f"${int(pet_deposit)}"
    if pd.isna(security_deposit) == True:
        security_deposit = 'Unknown'
    elif pd.isna(security_deposit) == False:
        security_deposit = f"${int(security_deposit)}"
    if pd.isna(other_deposit) == True:
        other_deposit = 'Unknown'
    elif pd.isna(other_deposit) == False:
        other_deposit = f"${int(other_deposit)}"
   # If there's no MLS photo, set it to an empty string so it doesn't display on the tooltip
   # Basically, the HTML block should just be an empty Img tag
    if pd.isna(mls_photo) == True:
        mls_photo_html_block = html.Img(
          src='',
          referrerPolicy='noreferrer',
          style={
            'display':'block',
            'width':'100%',
            'margin-left':'auto',
            'margin-right':'auto'
          },
          id='mls_photo_div'
        )
    # If there IS an MLS photo, just set it to itself
    # The HTML block should be an Img tag wrapped inside a parent <a href> tag so the image will be clickable
    elif pd.isna(mls_photo) == False:
        mls_photo_html_block = html.A( # wrap the Img inside a parent <a href> tag 
            html.Img(
              src=f'{mls_photo}',
              referrerPolicy='noreferrer',
              style={
                'display':'block',
                'width':'100%',
                'margin-left':'auto',
                'margin-right':'auto'
              },
              id='mls_photo_div'
            ),
          href=f"{mls_number_hyperlink}",
          referrerPolicy='noreferrer',
          target='_blank'
        )
    # Return the HTML snippet but NOT as a string. See https://github.com/thedirtyfew/dash-leaflet/issues/142#issuecomment-1157890463 
    return [
      html.Div([ # This is where the MLS photo will go (at the top and centered of the tooltip)
          mls_photo_html_block
      ]),
      html.Table([ # Create the table
        html.Tbody([ # Create the table body
          html.Tr([ # Start row #1
            html.Td("Listed Date"), html.Td(f"{listed_date}")
          ]), # end row #1
          html.Tr([ 
            html.Td("Street Address"), html.Td(f"{street_address}")
          ]),
          html.Tr([ 
            # Use a hyperlink to link to BHHS, don't use a referrer, and open the link in a new tab
            # https://www.freecodecamp.org/news/how-to-use-html-to-open-link-in-new-tab/
            html.Td(html.A("Listing ID (MLS#)", href="https://github.com/perfectly-preserved-pie/larentals/wiki#listing-id", target='_blank')), html.Td(html.A(f"{mls_number}", href=f"{mls_number_hyperlink}", referrerPolicy='noreferrer', target='_blank'))
          ]),
          html.Tr([ 
            html.Td("Rental Price"), html.Td(f"${lc_price}")
          ]),
          html.Tr([
            html.Td("Price Per Square Foot"), html.Td(f"{price_per_sqft}")
          ]),
          html.Tr([
            html.Td(html.A("Bedrooms/Bathrooms", href="https://github.com/perfectly-preserved-pie/larentals/wiki#bedroomsbathrooms", target='_blank')), html.Td(f"{brba}")
          ]),
          html.Tr([
            html.Td("Square Feet"), html.Td(f"{square_ft}")
          ]),
          html.Tr([
            html.Td("Year Built"), html.Td(f"{year}")
          ]),
          html.Tr([
            html.Td("Garage Spaces"), html.Td(f"{garage}"),
          ]),
          html.Tr([
            html.Td("Pets Allowed?"), html.Td(f"{pets}"),
          ]),
          html.Tr([
            html.Td("List Office Phone"), html.Td(f"{phone}"),
          ]),
          html.Tr([
            html.Td(html.A("Rental Terms", href="https://github.com/perfectly-preserved-pie/larentals/wiki#rental-terms", target='_blank')), html.Td(f"{terms}"),
          ]),
          html.Tr([
            html.Td("Furnished?"), html.Td(f"{furnished}"),
          ]),
          html.Tr([
            html.Td("Security Deposit"), html.Td(f"{security_deposit}"),
          ]),
          html.Tr([
            html.Td("Pet Deposit"), html.Td(f"{pet_deposit}"),
          ]),
          html.Tr([
            html.Td("Key Deposit"), html.Td(f"{key_deposit}"),
          ]),
          html.Tr([
            html.Td("Other Deposit"), html.Td(f"{other_deposit}"),
          ]),
          html.Tr([                                                                                            
            html.Td(html.A("Physical Sub Type", href="https://github.com/perfectly-preserved-pie/larentals/wiki#physical-sub-type", target='_blank')), html.Td(f"{sub_type}")                                                                                    
          ]), # end rows
        ]), # end body
      ]), # end table
    ]

# Create a function to get coordinates from the full street address
def return_coordinates(address):
    try:
        geocode_info = g.geocode(address)
        lat = float(geocode_info.latitude)
        lon = float(geocode_info.longitude)
        coords = f"{lat}, {lon}"
    except Exception as e:
        lat = NaN
        lon = NaN
        coords = NaN
        logging.warn(f"Couldn't fetch coordinates because of {e}.")
        pass
    logging.info(f"Fetched coordinates ({coords}) for {address}.")
    return lat, lon, coords

# Create a function to find missing postal codes based on short address
def return_postalcode(address):
    try:
        # Forward geocoding the short address so we can get coordinates
        geocode_info = g.geocode(address)
        # Reverse geocoding the coordinates so we can get the address object components
        components = g.geocode(f"{geocode_info.latitude}, {geocode_info.longitude}").raw['address_components']
        # Create a dataframe from this list of dictionaries
        components_df = pd.DataFrame(components)
        for row in components_df.itertuples():
            # Select the row that has the postal_code list
            if row.types == ['postal_code']:
                postalcode = row.long_name
    except Exception:
        postalcode = NaN
        pass
    logging.info(f"Fetched postal code {postalcode} for {address}.")
    return postalcode

## Webscraping Time
# Create a function to scrape the listing's Berkshire Hathaway Home Services (BHHS) page using BeautifulSoup 4 and extract some info
def webscrape_bhhs(url, row_index):
    try:
        response = requests.get(url)
        soup = bs4(response.text, 'html.parser')
        # Split the p class into strings and get the last element in the list
        # https://stackoverflow.com/a/64976919
        listed_date = soup.find('p', attrs={'class' : 'summary-mlsnumber'}).text.split()[-1]
        # Now find the URL for the "feature" photo of the listing
        photo = soup.find('a', attrs={'class' : 'show-listing-details'}).contents[1]['src']
        # Now find the URL to the actual listing instead of just the search result page
        link = 'https://www.bhhscalifornia.com' + soup.find('a', attrs={'class' : 'btn cab waves-effect waves-light btn-details show-listing-details'})['href']
    except AttributeError as e:
        listed_date = pd.NaT
        photo = NaN
        link = NaN
        logging.warn(f"Couldn't fetch some BHHS webscraping info because of {e}.")
        pass
    logging.info(f"Fetched Listed Date, MLS Photo, and BHHS link for row {row_index}...")
    return listed_date, photo, link

# Create a function to upload the file to ImageKit and then transform it
# https://github.com/imagekit-developer/imagekit-python#file-upload
def imagekit_transform(bhhs_url, mls):
    # if the MLS photo URL from BHHS isn't null (a photo IS available), then upload it to ImageKit
    if pd.isnull(bhhs_url) == False:
        try:
            uploaded_image = imagekit.upload_file(
                file= f"{bhhs_url}", # required
                file_name= f"{mls}.jpg", # required
                options= {
                    "is_private_file": False,
                    "use_unique_file_name": False,
                }
            )
        except Exception as e:
            logging.warning(f"Couldn't upload image to ImageKit because {e}. Passing on...")
            uploaded_image = 'ERROR'
            pass
    elif pd.isnull(bhhs_url) == True:
        uploaded_image = 'ERROR'
        logging.info(f"No image URL found. Not uploading anything to ImageKit.")
    # Now transform the uploaded image
    # https://github.com/imagekit-developer/imagekit-python#url-generation
    if 'ERROR' not in uploaded_image:
        try:
            global transformed_image
            transformed_image = imagekit.url({
                "src": f"{uploaded_image['response']['url']}",
                "transformation" : [{
                    "height": "300",
                    "width": "400"
                }]
            })
        except Exception as e:
            logging.warning(f"Couldn't transform image because {e}. Passing on...")
            transformed_image = None
            pass
    elif 'ERROR' in uploaded_image:
        logging.info(f"No image URL found. Not transforming anything.")
        transformed_image = None
    return transformed_image

# Filter the dataframe and return only rows with a NaN postal code
# For some reason some Postal Codes are "Assessor" :| so we need to include that string in an OR operation
# Then iterate through this filtered dataframe and input the right info we get using geocoding
for row in df.loc[(df['PostalCode'].isnull()) | (df['PostalCode'] == 'Assessor')].itertuples():
    missing_postalcode = return_postalcode(df.loc[(df['PostalCode'].isnull()) | (df['PostalCode'] == 'Assessor')].at[row.Index, 'Short Address'])
    df.at[row.Index, 'PostalCode'] = missing_postalcode

# Cast the PostalCode column as string
# Yes, postal codes are all integers but we're not doing any mathematical operations on them so a performance hit is irrelevant here
df['PostalCode'] = df['PostalCode'].astype(str)

# Now that we have street addresses and postal codes, we can put them together
# Create a new column with the full street address
# Also strip whitespace from the St Name column
df["Full Street Address"] = df["St#"] + ' ' + df["St Name"].str.strip() + ',' + ' ' + df['City'] + ' ' + df["PostalCode"]

# Iterate through the dataframe and get the listed date and photo for rows that don't have them
# If the Listed Date column is already present, iterate through the null cells
# We can use the presence of a Listed Date as a proxy for MLS Photo; generally, either both or neither exist/don't exist together
# This assumption will reduce the number of HTTP requests we send to BHHS
if 'Listed Date' in df.columns:
    for row in df.loc[df['Listed Date'].isnull()].itertuples():
        mls_number = row[1]
        webscrape = webscrape_bhhs(f"https://www.bhhscalifornia.com/for-lease/{mls_number}-t_q;/", {row.Index})
        df.at[row.Index, 'Listed Date'] = webscrape[0]
        df.at[row.Index, 'MLS Photo'] = imagekit_transform(webscrape[1], row._1)
        df.at[row.Index, 'bhhs_url'] = webscrape[2]
# if the Listed Date column doesn't exist (i.e this is a first run), create it using df.at
elif 'Listed Date' not in df.columns:
    for row in df.itertuples():
        mls_number = row[1]
        webscrape = webscrape_bhhs(f"https://www.bhhscalifornia.com/for-lease/{mls_number}-t_q;/", {row.Index})
        df.at[row.Index, 'Listed Date'] = webscrape[0]
        df.at[row.Index, 'MLS Photo'] = imagekit_transform(webscrape[1], row._1)
        df.at[row.Index, 'bhhs_url'] = webscrape[2]

# Iterate through the dataframe and fetch coordinates for rows that don't have them
# If the Coordinates column is already present, iterate through the null cells
# Similiar to above, we can use the presence of the Coordinates column as a proxy for Longitude and Latitude; all 3 should exist together or none at all
# This assumption will reduce the number of API calls to Google Maps
if 'Coordinates' in df.columns:
    for row in df['Coordinates'].isnull().itertuples():
        coordinates = return_coordinates(df.at[row.Index, 'Full Street Address'])
        df.at[row.Index, 'Latitude'] = coordinates[0]
        df.at[row.Index, 'Longitude'] = coordinates[1]
        df.at[row.Index, 'Coordinates'] = coordinates[2]
# If the Coordinates column doesn't exist (i.e this is a first run), create it using df.at
elif 'Coordinates' not in df.columns:
    for row in df.itertuples():
        coordinates = return_coordinates(df.at[row.Index, 'Full Street Address'])
        df.at[row.Index, 'Latitude'] = coordinates[0]
        df.at[row.Index, 'Longitude'] = coordinates[1]
        df.at[row.Index, 'Coordinates'] = coordinates[2]

# Remove all $ and , symbols from specific columns
# https://stackoverflow.com/a/46430853
cols = ['DepositKey', 'DepositOther', 'DepositPets', 'DepositSecurity', 'List Price', 'Price Per Square Foot', 'Sqft']
# pass them to df.replace(), specifying each char and it's replacement:
df[cols] = df[cols].replace({'\$': '', ',': ''}, regex=True)

# Split the Bedroom/Bathrooms column into separate columns based on delimiters
# Based on the example given in the spreadsheet: 2 (beds) / 1 (total baths),1 (full baths) ,0 (half bath), 0 (three quarter bath)
# Realtor logic based on https://www.realtor.com/advice/sell/if-i-take-out-the-tub-does-a-bathroom-still-count-as-a-full-bath/
# TIL: A full bathroom is made up of four parts: a sink, a shower, a bathtub, and a toilet. Anything less than that, and you can’t officially consider it a full bath.
df['Bedrooms'] = df['Br/Ba'].str.split('/', expand=True)[0]
df['Total Bathrooms'] = (df['Br/Ba'].str.split('/', expand=True)[1]).str.split(',', expand=True)[0]
df['Full Bathrooms'] = (df['Br/Ba'].str.split('/', expand=True)[1]).str.split(',', expand=True)[1]
df['Half Bathrooms'] = (df['Br/Ba'].str.split('/', expand=True)[1]).str.split(',', expand=True)[2]
df['Three Quarter Bathrooms'] = (df['Br/Ba'].str.split('/', expand=True)[1]).str.split(',', expand=True)[3]

# Remove the square footage & YrBuilt abbreviations
df['Sqft'] = df['Sqft'].str.split('/').str[0]
df['YrBuilt'] = df['YrBuilt'].str.split('/').str[0]

# Convert a few columns into integers 
# To prevent weird TypeError shit like TypeError: '>=' not supported between instances of 'str' and 'int'
df['List Price'] = df['List Price'].apply(pd.to_numeric)
df['Bedrooms'] = df['Bedrooms'].apply(pd.to_numeric)
df['Total Bathrooms'] = df['Total Bathrooms'].apply(pd.to_numeric)
df['PostalCode'] = df['PostalCode'].apply(pd.to_numeric, errors='coerce') # convert non-integers into NaNs
df['Sqft'] = df['Sqft'].apply(pd.to_numeric, errors='coerce') # convert non-integers into NaNs
df['YrBuilt'] = df['YrBuilt'].apply(pd.to_numeric, errors='coerce') # convert non-integers into NaNs
df['Price Per Square Foot'] = df['Price Per Square Foot'].apply(pd.to_numeric, errors='coerce') # convert non-integers into NaNs
df['Garage Spaces'] = df['Garage Spaces'].apply(pd.to_numeric, errors='coerce') # convert non-integers into NaNs
df['Latitude'] = df['Latitude'].apply(pd.to_numeric, errors='coerce') # convert non-integers into NaNs
df['Longitude'] = df['Longitude'].apply(pd.to_numeric, errors='coerce') # convert non-integers into NaNs
df['DepositKey'] = df['DepositKey'].apply(pd.to_numeric, errors='coerce')
df['DepositOther'] = df['DepositOther'].apply(pd.to_numeric, errors='coerce')
df['DepositPets'] = df['DepositPets'].apply(pd.to_numeric, errors='coerce')
df['DepositSecurity'] = df['DepositSecurity'].apply(pd.to_numeric, errors='coerce')

# Convert the listed date into DateTime and set missing values to be NaT
# Infer datetime format for faster parsing
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html
df['Listed Date'] = pd.to_datetime(df['Listed Date'], errors='coerce', infer_datetime_format=True)

# Per CA law, ANY type of deposit is capped at rent * 3 months
# It doesn't matter the type of deposit, they all have the same cap
# Despite that, some landlords/realtors will list the property with an absurd deposit (100k? wtf) so let's rewrite those
# Use numpy .values to rewrite anything greater than $18000 ($6000 rent * 3 months) into $18000
# https://stackoverflow.com/a/54426197
df['DepositSecurity'].values[df['DepositSecurity'] > 18000] = 18000
df['DepositPets'].values[df['DepositPets'] > 18000] = 18000
df['DepositOther'].values[df['DepositOther'] > 18000] = 18000
df['DepositKey'].values[df['DepositKey'] > 18000] = 18000

# Tag each row with the date it was generated
for row in df.itertuples():
  if 'date_generated' in df.columns:
    pass
  elif 'date_generated' not in df.columns:
    df.at[row.Index, 'date_generated'] = datetime.now().date()

# The rental marker is hot and properties go off market fast
# Keep all rows less than a "month" old (31 days)
df['date_generated'] = df['date_generated'] >= date.today() - timedelta(31)

# Keep rows with less than 6 bedrooms
# 6 bedrooms and above are probably multi family investments and not actual rentals
# They also skew the outliers, causing the sliders to go way up
df = df[df.Bedrooms < 6]

# Reindex the dataframe
df.reset_index(drop=True, inplace=True)

# Define HTML code for the popup so it looks pretty and nice
def popup_html(row):
    i = row.Index
    street_address=df['Full Street Address'].at[i] 
    mls_number=df['Listing ID (MLS#)'].at[i]
    mls_number_hyperlink=df['bhhs_url'].at[i]
    mls_photo = df['MLS Photo'].at[i]
    lc_price = df['List Price'].at[i] 
    price_per_sqft=df['Price Per Square Foot'].at[i]                  
    brba = df['Br/Ba'].at[i]
    square_ft = df['Sqft'].at[i]
    year = df['YrBuilt'].at[i]
    garage = df['Garage Spaces'].at[i]
    pets = df['PetsAllowed'].at[i]
    phone = df['List Office Phone'].at[i]
    terms = df['Terms'].at[i]
    sub_type = df['Sub Type'].at[i]
    listed_date = pd.to_datetime(df['Listed Date'].at[i]).date() # Convert the full datetime into date only. See https://stackoverflow.com/a/47388569
    furnished = df['Furnished'].at[i]
    key_deposit = df['DepositKey'].at[i]
    other_deposit = df['DepositOther'].at[i]
    pet_deposit = df['DepositPets'].at[i]
    security_deposit = df['DepositSecurity'].at[i]
    # If there's no square footage, set it to "Unknown" to display for the user
    # https://towardsdatascience.com/5-methods-to-check-for-nan-values-in-in-python-3f21ddd17eed
    if pd.isna(square_ft) == True:
        square_ft = 'Unknown'
    # If there IS a square footage, convert it into an integer (round number)
    elif pd.isna(square_ft) == False:
        square_ft = f"{int(square_ft)} sq. ft"
    # Repeat above for Year Built
    if pd.isna(year) == True:
        year = 'Unknown'
    # If there IS a square footage, convert it into an integer (round number)
    elif pd.isna(year) == False:
        year = f"{int(year)}"
    # Repeat above for garage spaces
    if pd.isna(garage) == True:
        garage = 'Unknown'
    elif pd.isna(garage) == False:
        garage = f"{int(garage)}"
    # Repeat for ppsqft
    if pd.isna(price_per_sqft) == True:
        price_per_sqft = 'Unknown'
    elif pd.isna(price_per_sqft) == False:
        price_per_sqft = f"${float(price_per_sqft)}"
    # Repeat for listed date
    if pd.isna(listed_date) == True:
        listed_date = 'Unknown'
    elif pd.isna(listed_date) == False:
        listed_date = f"{listed_date}"
    # Repeat for furnished
    if pd.isna(furnished) == True:
        furnished = 'Unknown'
    elif pd.isna(furnished) == False:
        furnished = f"{furnished}"
    # Repeat for the deposits
    if pd.isna(key_deposit) == True:
        key_deposit = 'Unknown'
    elif pd.isna(key_deposit) == False:
        key_deposit = f"${int(key_deposit)}"
    if pd.isna(pet_deposit) == True:
        pet_deposit = 'Unknown'
    elif pd.isna(pet_deposit) == False:
        pet_deposit = f"${int(pet_deposit)}"
    if pd.isna(security_deposit) == True:
        security_deposit = 'Unknown'
    elif pd.isna(security_deposit) == False:
        security_deposit = f"${int(security_deposit)}"
    if pd.isna(other_deposit) == True:
        other_deposit = 'Unknown'
    elif pd.isna(other_deposit) == False:
        other_deposit = f"${int(other_deposit)}"
   # If there's no MLS photo, set it to an empty string so it doesn't display on the tooltip
   # Basically, the HTML block should just be an empty Img tag
    if pd.isna(mls_photo) == True:
        mls_photo_html_block = html.Img(
          src='',
          referrerPolicy='noreferrer',
          style={
            'display':'block',
            'width':'100%',
            'margin-left':'auto',
            'margin-right':'auto'
          },
          id='mls_photo_div'
        )
    # If there IS an MLS photo, just set it to itself
    # The HTML block should be an Img tag wrapped inside a parent <a href> tag so the image will be clickable
    elif pd.isna(mls_photo) == False:
        mls_photo_html_block = html.A( # wrap the Img inside a parent <a href> tag 
            html.Img(
              src=f'{mls_photo}',
              referrerPolicy='noreferrer',
              style={
                'display':'block',
                'width':'100%',
                'margin-left':'auto',
                'margin-right':'auto'
              },
              id='mls_photo_div'
            ),
          href=f"{mls_number_hyperlink}",
          referrerPolicy='noreferrer',
          target='_blank'
        )
    # Return the HTML snippet but NOT as a string. See https://github.com/thedirtyfew/dash-leaflet/issues/142#issuecomment-1157890463 
    return [
      html.Div([ # This is where the MLS photo will go (at the top and centered of the tooltip)
          mls_photo_html_block
      ]),
      html.Table([ # Create the table
        html.Tbody([ # Create the table body
          html.Tr([ # Start row #1
            html.Td("Listed Date"), html.Td(f"{listed_date}")
          ]), # end row #1
          html.Tr([ 
            html.Td("Street Address"), html.Td(f"{street_address}")
          ]),
          html.Tr([ 
            # Use a hyperlink to link to BHHS, don't use a referrer, and open the link in a new tab
            # https://www.freecodecamp.org/news/how-to-use-html-to-open-link-in-new-tab/
            html.Td(html.A("Listing ID (MLS#)", href="https://github.com/perfectly-preserved-pie/larentals/wiki#listing-id", target='_blank')), html.Td(html.A(f"{mls_number}", href=f"{mls_number_hyperlink}", referrerPolicy='noreferrer', target='_blank'))
          ]),
          html.Tr([ 
            html.Td("Rental Price"), html.Td(f"${lc_price}")
          ]),
          html.Tr([
            html.Td("Price Per Square Foot"), html.Td(f"{price_per_sqft}")
          ]),
          html.Tr([
            html.Td(html.A("Bedrooms/Bathrooms", href="https://github.com/perfectly-preserved-pie/larentals/wiki#bedroomsbathrooms", target='_blank')), html.Td(f"{brba}")
          ]),
          html.Tr([
            html.Td("Square Feet"), html.Td(f"{square_ft}")
          ]),
          html.Tr([
            html.Td("Year Built"), html.Td(f"{year}")
          ]),
          html.Tr([
            html.Td("Garage Spaces"), html.Td(f"{garage}"),
          ]),
          html.Tr([
            html.Td("Pets Allowed?"), html.Td(f"{pets}"),
          ]),
          html.Tr([
            html.Td("List Office Phone"), html.Td(f"{phone}"),
          ]),
          html.Tr([
            html.Td(html.A("Rental Terms", href="https://github.com/perfectly-preserved-pie/larentals/wiki#rental-terms", target='_blank')), html.Td(f"{terms}"),
          ]),
          html.Tr([
            html.Td("Furnished?"), html.Td(f"{furnished}"),
          ]),
          html.Tr([
            html.Td("Security Deposit"), html.Td(f"{security_deposit}"),
          ]),
          html.Tr([
            html.Td("Pet Deposit"), html.Td(f"{pet_deposit}"),
          ]),
          html.Tr([
            html.Td("Key Deposit"), html.Td(f"{key_deposit}"),
          ]),
          html.Tr([
            html.Td("Other Deposit"), html.Td(f"{other_deposit}"),
          ]),
          html.Tr([                                                                                            
            html.Td(html.A("Physical Sub Type", href="https://github.com/perfectly-preserved-pie/larentals/wiki#physical-sub-type", target='_blank')), html.Td(f"{sub_type}")                                                                                    
          ]), # end rows
        ]), # end body
      ]), # end table
    ]

# Iterate through and generate the HTML code
if 'popup_html' in df.columns:
    for row in df['popup_html'].isnull().itertuples():
        df.at[row.Index, 'popup_html'] = popup_html(row)
# If the popup_html column doesn't exist (i.e this is a first run), create it using df.at
elif 'popup_html' not in df.columns:
    df['popup_html'] = ''
    for row in df.itertuples():
        df.at[row.Index, 'popup_html'] = popup_html(row)

# Pickle the dataframe for later ingestion by app.py
# https://www.youtube.com/watch?v=yYey8ntlK_E
# Depending if a pickle file exists already, either create a new one or append the dataframe to an existing pickle file
path = './dataframe.pickle'
if exists(path) == False:
  df.to_pickle("dataframe.pickle")
elif exists(path) == True:
  # Load the old dataframe into memory
  df_old = pd.read_pickle("dataframe.pickle")
  # Combine both old and new dataframes
  df_new = pd.concat(df, df_old)
  # Pickle the new dataframe
  df_new.to_pickle("dataframe.pickle")