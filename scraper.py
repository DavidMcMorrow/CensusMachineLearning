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

search = driver.find_element_by_id("census_year")
search.send_keys("1901")
search.send_keys(Keys.RETURN)
search.send_keys(Keys.RETURN)
print("111111111")
search = driver.find_element_by_id("county19011911")
search.send_keys("Kerry")
search.send_keys(Keys.RETURN)
search.send_keys(Keys.RETURN)
print("2222222222")
search = driver.find_element_by_id("age")
search.send_keys("23")
search.send_keys(Keys.RETURN)
#search.send_keys(Keys.RETURN)
print("333333333")
search = driver.find_element_by_name("search")
#search.click()
#search.send_keys(Keys.RETURN)
print("HHHHHHEEEEEEEEEERRRRRRRRRRREEEEEEEEE")
print("HERE", driver.title)
try:
    results = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "breadcrumb"))
    )
    print("Results = ", results.text)
except:
    #sleep(15)
    driver.quit()


# search = driver.find_element_by_id("show_all")
# search.send_keys(Keys.RETURN)

# search = driver.find_element_by_class("next")
# search.send_keys(Keys.RETURN)



#driver.quit()
