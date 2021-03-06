from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

PATH = "C:\Program Files (x86)\chromedriver.exe"
driver = webdriver.Chrome(PATH)


driver.get("http://www.census.nationalarchives.ie/search/")

print(driver.title)
#driver.maximize_window()

search = driver.find_element_by_id("census_year")
search.send_keys("1851")
search.send_keys(Keys.RETURN)
search.send_keys(Keys.RETURN)

# search = driver.find_element_by_id("county1851")
# search.send_keys("Kerry")
# search.send_keys(Keys.RETURN)
# search.send_keys(Keys.RETURN)

search = driver.find_element_by_id("age")
search.send_keys("23")
search.send_keys(Keys.RETURN)

search = driver.find_element_by_name("search")
headings = []
people = []
try:
    for page in range(1, 1000):
        
        results = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.ID, "show_all"))
        )
        results.click()
        
        if page == 1:
            for j in range(1, 16):
                xpath = "//table/thead/tr/th[" + str(j) + "]"
                #print("xpath", xpath)
                header = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, xpath))
                )
                #print("headings", header.text)
                headings.append(header.text)
        #print("headings", headings)
        for rows in range(1, 11):
            person = []
            for cols in range(1, 16):
                cellPath = "//table/tbody/tr[" + str(rows) + "]/td[" + str(cols)+ "]"
                cells = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, cellPath))
                )
                #print("cells", cells.text)
                person.append(cells.text)
            people.append(person)
        #show100 = WebDriverWait(driver, 10).until(
        #  EC.presence_of_element_located((By.CLASS_NAME, "current"))
        #)
        #print("show100", show100)
        #show100.click()
        next10 = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "next"))
        )
        next10.click()
    
except:
    
    print("In Except")
    #driver.quit()
print("headings", headings)
print("people", len(people))
# search = driver.find_element_by_id("show_all")
# search.send_keys(Keys.RETURN)

# search = driver.find_element_by_class("next")
# search.send_keys(Keys.RETURN)

import csv
with open('test_1851.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(people)

#driver.quit()
