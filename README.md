# WhereToLive.LA

This is an interactive map based on /u/WilliamMcCarty's weekly spreadsheets of new rental listings in the /r/LArentals subreddit. Just like the actual spreadsheet, you can filter the map based on different criteria.

Some additional capabilities are offered, such as a featured MLS photo for the property and a link to the associated MLS listing page (if available).

## The Tech Stack
* [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) (webscraping MLS photos and links)
*    [Dash Leaflet](https://dash-leaflet.herokuapp.com/) (displaying the map and graphing the markers)
*    [Dash Bootstrap Components](https://dash-bootstrap-components.opensource.faculty.ai/) (the website layout and icons)
*    [GeoPy](https://geopy.readthedocs.io/en/stable/) (geocoding coordinates via the Google Maps API)
*    [ImageKit](https://github.com/imagekit-developer/imagekit-python) (resizing MLS photos into a standard size on the fly)
*    [Pandas](https://pandas.pydata.org/) (handling and manipulating the rental property data for each address)