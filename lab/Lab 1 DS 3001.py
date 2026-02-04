import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from bs4 import BeautifulSoup as soup # HTML parser
import requests # Page requests
import re # Regular expressions
import time # Time delays
import random # Random numbers

import requests
from bs4 import BeautifulSoup as soup
import re
import time
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------
# Step 1: Setup
# -----------------------
header = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0'
}
url = "https://charlottesville.craigslist.org/search/mca?purveyor=owner"

raw = requests.get(url, headers=header)
bsObj = soup(raw.content, 'html.parser')
listings = bsObj.find_all(class_="cl-static-search-result")

# -----------------------
# Step 2: Prepare lists for data
# -----------------------
data = []
links = []

# Define some motorcycle-related keywords for categorization
brands = [
    'honda','harley','yamaha','suzuki','kawasaki','bmw','ducati','triumph','ktm','royal enfield',
    'kawasaki','zero','indian','aprilia'
]
types = [
    'cruiser','sport','standard','touring','dirt','adventure','bobber','chopper','scooter','moped'
]

# -----------------------
# Step 3: Scrape main page listings
# -----------------------
for listing in listings:
    title = listing.find('div', class_='title').get_text().lower()
    price = listing.find('div', class_='price').get_text() if listing.find('div',class_='price') else np.nan
    link = listing.find(href=True)['href']
    links.append(link)

    # Brand extraction
    words = title.split()
    brand_hits = [w for w in words if w in brands]
    brand = brand_hits[0] if brand_hits else 'other'

    # Type extraction
    type_hits = [w for w in words if w in types]
    bike_type = type_hits[0] if type_hits else 'other'

    # Year extraction
    regex_year = re.search(r'19[0-9]{2}|20[0-9]{2}', title)
    year = int(regex_year.group(0)) if regex_year else np.nan

    data.append({
        'title': title,
        'price_text': price,
        'brand': brand,
        'type': bike_type,
        'year': year,
        'link': link
    })

df = pd.DataFrame(data)

# -----------------------
# Step 4: Clean price
# -----------------------
df['price'] = df['price_text'].astype(str).str.replace('$','', regex=False).str.replace(',','', regex=False)
df['price'] = pd.to_numeric(df['price'], errors='coerce')

# Keep only positive prices
df = df[df['price'] > 0].copy()

# -----------------------
# Step 5: Scrape detailed listings
# -----------------------
detail_records = []

for link in df['link']:
    time.sleep(random.randint(1, 3))
    raw = requests.get(link, headers=header)
    bs = soup(raw.content, 'html.parser')

    # Description
    try:
        text = bs.find(id='postingbody').get_text().replace('\n','').replace('QR Code Link to This Post','')
    except:
        text = ''

    # Condition
    try:
        condition = bs.find(class_='attr condition').find(href=True).get_text()
    except:
        condition = 'missing'

    # Mileage
    try:
        miles = bs.find(class_='attr auto_miles').find(class_='valu').get_text()
        miles = miles.replace(',','')
        miles = int(miles)
    except:
        miles = np.nan

    detail_records.append({
        'link': link,
        'description': text,
        'condition': condition,
        'miles': miles
    })

df_detail = pd.DataFrame(detail_records)

df = pd.merge(df, df_detail, on='link', how='left')

# -----------------------
# Step 6: Add age
# -----------------------
df['age'] = 2025 - df['year']

# -----------------------
# Step 7: Save CSV
# -----------------------
df.to_csv('craigslist_cville_motorcycles.csv', index=False)

# -----------------------
# Step 8: EDA & Plots
# -----------------------

# Summary stats
print("Price summary")
print(df['price'].describe())

print("\nAge summary")
print(df['age'].describe())

# Histogram: Price
plt.figure(figsize=(8,5))
df['price'].hist(grid=False)
plt.xlabel('Price')
plt.ylabel('Count')
plt.title('Motorcycle Prices')
plt.show()

# Histogram: Age
plt.figure(figsize=(8,5))
df['age'].hist(grid=False)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Motorcycle Age Distribution')
plt.show()

# Boxplot: Price by Brand
plt.figure(figsize=(10,6))
sns.boxplot(data=df, x='brand', y='price')
plt.xticks(rotation=45)
plt.title('Price by Brand')
plt.show()

# Boxplot: Price by Type
plt.figure(figsize=(10,6))
sns.boxplot(data=df, x='type', y='price')
plt.xticks(rotation=45)
plt.title('Price by Motorcycle Type')
plt.show()

# Scatter: Age vs Price
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='age', y='price', hue='brand')
plt.title('Age vs Price by Brand')
plt.show()

# Log transformations
df['log_price'] = np.log(df['price'])
df['log_age'] = np.log(df['age'].replace(0, np.nan))

# Correlation & covariance
print("Covariance (log_price, log_age):")
print(df[['log_price','log_age']].cov())

print("Correlation (log_price, log_age):")
print(df[['log_price','log_age']].corr())

# Joint plot: log log
sns.jointplot(data=df, x='log_age', y='log_price', kind='scatter')
plt.show()

