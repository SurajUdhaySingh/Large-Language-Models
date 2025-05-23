import requests
from bs4 import BeautifulSoup
import os
import time

def download_wiki_articles(article_titles, output_file="wiki_articles.txt"):
    """
    Downloads Wikipedia articles from a list of titles and saves their text to a file
    with minimal extra newlines.

    Args:
        article_titles (list): List of Wikipedia article titles
        output_file (str): Name of the output text file (default: wiki_articles.txt)
    """
    # Open file in write mode to start fresh (overwrites existing file)
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, title in enumerate(article_titles):
            try:
                # Format Wikipedia URL
                url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                
                # Send HTTP request
                response = requests.get(url)
                response.raise_for_status()
                
                # Parse HTML content
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find the main content div
                content = soup.find('div', {'id': 'mw-content-text'})
                if not content:
                    print(f"Could not find content for {title}")
                    continue
                
                # Get all paragraphs in the main content
                paragraphs = content.find_all('p')
                
                # Write article title as header (only add newline if not first article)
                if i > 0:
                    f.write("\n")
                f.write(f"=== {title} ===\n")
                
                # Extract and write text from each paragraph, avoiding extra newlines
                for j, para in enumerate(paragraphs):
                    text = para.get_text().strip()
                    if text:  # Only write non-empty paragraphs
                        f.write(text)
                        # Add single newline after each paragraph, except the last one
                        if j < len(paragraphs) - 1:
                            f.write("\n")
                
                print(f"Successfully downloaded {title}")
                
            except requests.RequestException as e:
                print(f"Error downloading {title}: {e}")
            except Exception as e:
                print(f"Error processing {title}: {e}")
            
            # Add a small delay to be polite to Wikipedia servers
            time.sleep(1)
    
    print(f"\nAll articles processed. Output saved to {output_file}")

