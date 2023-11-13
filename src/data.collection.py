import csv
import re
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import chromedriver_autoinstaller
import time

# Install ChromeDriver if not installed and set up in PATH
chromedriver_autoinstaller.install()

# IMDb movie URL 
url = "https://www.imdb.com/title/tt1375666/reviews/?ref_=tt_ql_2S"

# Initialize a Chrome WebDriver
driver = webdriver.Chrome()

# Send an HTTP GET request to the IMDb page
driver.get(url)

# Find the "Load More" button and click it repeatedly until you have 500 reviews
max_reviews = 500
scraped_data = []

while len(scraped_data) < max_reviews:
    try:
        load_more_button = driver.find_element(By.ID, 'load-more-trigger')
        
        # Wait until the button is clickable
        wait = WebDriverWait(driver, 10)  # Adjust the timeout as needed
        wait.until(EC.element_to_be_clickable((By.ID, 'load-more-trigger')))
        
        # Scroll the button into view
        driver.execute_script("arguments[0].scrollIntoView();", load_more_button)
        
        # Click the button
        load_more_button.click()

        # Wait for the new reviews to load
        time.sleep(5)  # Adjust the time as needed
        
    except Exception as e:
        break

# Once all reviews are loaded, parse the page using BeautifulSoup
soup = BeautifulSoup(driver.page_source, 'html.parser')
review_elements = soup.find_all('div', class_='text show-more__control')
for review_element in review_elements:
    review_text = review_element.get_text(strip=True)
    scraped_data.append({"review_text": review_text})

# Close the WebDriver
driver.quit()

# Create a Pandas DataFrame
df = pd.DataFrame(scraped_data)

# Data preprocessing
def preprocess_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Strip leading and trailing whitespace
    text = text.strip()
    
    return text

# Apply data preprocessing to the "review_text" column
df['review_text'] = df['review_text'].apply(preprocess_text)

# Deduplicate reviews
df.drop_duplicates(subset=['review_text'], keep='first', inplace=True)

# Handle missing data
df.dropna(subset=['review_text'], inplace=True)

# Only keep the first 500 reviews
df = df.head(max_reviews)

# Specify the CSV file path
csv_file = "data/movie_reviews.csv"

# Export the cleaned data to a CSV file
df.to_csv(csv_file, index=False, encoding="utf-8")

print(f"Scraped and cleaned {len(df)} reviews, and saved to {csv_file}")