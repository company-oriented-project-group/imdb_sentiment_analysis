import csv
import requests
from bs4 import BeautifulSoup

# IMDb movie URL 
url = "https://www.imdb.com/title/tt1375666/reviews/?ref_=tt_ql_2S"

# Send an HTTP GET request to the IMDb page
response = requests.get(url)

if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract movie reviews, ratings, or other relevant data from the page
   
    scraped_data = []

    review_elements = soup.find_all('div', class_='text show-more__control')
    for review_element in review_elements:
        review_text = review_element.get_text(strip=True)
        scraped_data.append({"review_text": review_text})
    
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

else:
    print("Failed to retrieve the webpage. Status code:", response.status_code)