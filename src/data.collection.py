import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import chromedriver_autoinstaller

# Install ChromeDriver if not installed and set up in PATH
chromedriver_autoinstaller.install()

# IMDb movie URL 
url = "https://www.imdb.com/title/tt1375666/reviews/?ref_=tt_ql_2S"

# Initialize a Chrome WebDriver
driver = webdriver.Chrome()

# Send an HTTP GET request to the IMDb page
driver.get(url)

# Find the "Load More" button and click it until you have 500 reviews
max_reviews = 500
scraped_data = []

while len(scraped_data) < max_reviews:
    load_more_button = driver.find_element(By.ID, 'load-more-trigger')
    
    # Wait until the button is clickable
    wait = WebDriverWait(driver, 10)  # Adjust the timeout as needed
    wait.until(EC.element_to_be_clickable((By.ID, 'load-more-trigger')))
    
    # Scroll the button into view
    driver.execute_script("arguments[0].scrollIntoView();", load_more_button)
    
    # Click the button
    load_more_button.click()

    # Wait for the new reviews to load 
    driver.implicitly_wait(5)

    # Once all reviews are loaded, parse the page using BeautifulSoup
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    review_elements = soup.find_all('div', class_='text show-more__control')
    for review_element in review_elements:
        review_text = review_element.get_text(strip=True)
        scraped_data.append({"review_text": review_text})

    if len(scraped_data) >= max_reviews:
        break

# Close the WebDriver
driver.quit()

# Only keep the first 500 reviews
scraped_data = scraped_data[:max_reviews]

# Specify the CSV file path
csv_file = "movie_reviews.csv"

# Create or open the CSV file for writing
with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
    # Define the CSV writer
    writer = csv.writer(file)

    # Write the header row
    writer.writerow(["review_text"])

    # Write the data rows
    for data in scraped_data:
        writer.writerow([data["review_text"]])

print(f"Scraped {len(scraped_data)} reviews and saved to {csv_file}")