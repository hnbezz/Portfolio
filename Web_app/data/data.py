import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.colors
from collections import OrderedDict
import requests


# default list of all countries of interest
country_default = OrderedDict([('Canada', 'CAN'), ('United States', 'USA'), 
  ('Brazil', 'BRA'), ('France', 'FRA'), ('India', 'IND'), ('Italy', 'ITA'), 
  ('Germany', 'DEU'), ('United Kingdom', 'GBR'), ('China', 'CHN'), ('Japan', 'JPN')])


def return_figures(countries=country_default):
  """Creates four plotly visualizations using the World Bank API
  # Example of the World Bank API endpoint:
  # arable land for the United States and Brazil from 1990 to 2015
  # http://api.worldbank.org/v2/countries/usa;bra/indicators/AG.LND.ARBL.HA?date=1990:2015&per_page=1000&format=json
    Args:
        country_default (dict): list of countries for filtering the data
    Returns:
        list (dict): list containing the four plotly visualizations
  """

  # when the countries variable is empty, use the country_default dictionary
  if not bool(countries):
    countries = country_default

  # prepare filter data for World Bank API
  # the API uses ISO-3 country codes separated by ;
  country_filter = list(countries.values())
  country_filter = [x.lower() for x in country_filter]
  country_filter = ';'.join(country_filter)

  # World Bank indicators of interest for pulling data
  indicators2 = ['AG.LND.ARBL.HA.PC', 'SP.RUR.TOTL.ZS', 'SP.RUR.TOTL.ZS', 'AG.LND.FRST.ZS']
  indicators = ['SP.URB.TOTL', 'SP.POP.SCIE.RD.P6','SL.TLF.ADVN.ZS', 'SL.UEM.ADVN.ZS'] # urban population, Researchers in R&D (per million people),  Labor force with advanced education, Unemployment with advanced education (% of total labor force with advanced education)
    
  data_frames = [] # stores the data frames with the indicator data of interest
  urls = [] # url endpoints for the World Bank API

  # pull data from World Bank API and clean the resulting json
  # results stored in data_frames variable
  for indicator in indicators:
    url = 'http://api.worldbank.org/v2/countries/' + country_filter + '/indicators/' + indicator + '?date=1990:2015&per_page=1000&format=json'
    urls.append(url)

    try:
      r = requests.get(url)
      data = r.json()[1]
    except:
      print('could not load data ', indicator)

    for i, value in enumerate(data):
      value['indicator'] = value['indicator']['value']
      value['country'] = value['country']['value']

    data_frames.append(data)
  
  # first chart plots arable land from 1990 to 2015 in top 10 economies 
  # as a line chart
  graph_one = []
  df_one = pd.DataFrame(data_frames[0])

  # filter and sort values for the visualization
  # filtering plots the countries in decreasing order by their values
  df_one = df_one[df_one['date'] > '1990']
  df_one = df_one[df_one['date'] < '2018']
  df_one.sort_values('value', ascending=False, inplace=True)

  # this  country list is re-used by all the charts to ensure legends have the same
  # order and color
  countrylist = df_one.country.unique().tolist()
  
  for country in countrylist:
      x_val = df_one[df_one['country'] == country].date.tolist()
      y_val =  df_one[df_one['country'] == country].value.tolist()
      graph_one.append(
          go.Scatter(
          x = x_val,
          y = y_val,
          mode = 'lines+markers',
          name = country
          )
      )

  layout_one = dict(title = 'Urban population from 1990 to 2015',
                xaxis = dict(title = 'Year',
                  autotick=True),
                yaxis = dict(title = 'Urban Population'),
                )

  # second chart plots Researchers in R&D (per million people) in a bar chart 
  graph_two = []
  df_two = pd.DataFrame(data_frames[1])
  df_two.sort_values('value', ascending=False, inplace=True)
  df_two = df_two[df_two['date'] == '2015'] 

  graph_two.append(
      go.Bar(
      x = df_two.country.tolist(),
      y = df_two.value.tolist(),
      )
  )

  layout_two = dict(title = 'Researchers in R&D (per million people) in 2015',
                xaxis = dict(title = 'Country',),
                yaxis = dict(title = 'Researchers in R&D (per million people)'),
                )

  # third chart plots Labor force with advanced education from 1990 to 2018
  graph_three = []
  df_three = pd.DataFrame(data_frames[2])
  df_three = df_three[df_three['date'] > '1990']
  df_three = df_three[df_three['date'] < '2018']

  df_three.sort_values('date', ascending=False, inplace=True)
  for country in countrylist:
      x_val = df_three[df_three['country'] == country].date.tolist()
      y_val =  df_three[df_three['country'] == country].value.tolist()
      graph_three.append(
          go.Scatter(
          x = x_val,
          y = y_val,
          mode = 'lines+markers',
          name = country
          )
      )

  layout_three = dict(title = 'Labor force with advanced education from 1990 to 2015',
                xaxis = dict(title = 'Year',
                  autotick=True),
                yaxis = dict(title = 'Labor force with advanced education'),
                )

  # fourth chart shows Unemployment with advanced education (% of total labor force with advanced education)
  graph_four = []
  df_four = pd.DataFrame(data_frames[3])
  df_four.sort_values('value', ascending=False, inplace=True)
  df_four = df_four[df_four['date'] == '2015'] 

  graph_four.append(
      go.Bar(
      x = df_four.country.tolist(),
      y = df_four.value.tolist(),
      )
  )

  layout_four = dict(title = 'Unemployment with adv. ed. (% of total) in 2015',
                xaxis = dict(title = 'Country',),
                yaxis = dict(title = 'Unemp. with adv. edudcation (% of total)'),
                )


  # append all charts
  figures = []
  figures.append(dict(data=graph_one, layout=layout_one))
  figures.append(dict(data=graph_two, layout=layout_two))
  figures.append(dict(data=graph_three, layout=layout_three))
  figures.append(dict(data=graph_four, layout=layout_four))

  return figures