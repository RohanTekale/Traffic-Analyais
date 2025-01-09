# import streamlit as st
# import pandas as pd
# import numpy as np
# from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
# from sklearn.preprocessing import StandardScaler
# import plotly.express as px
# from PIL import Image
#
# # Load and preprocess data
# @st.cache(allow_output_mutation=True)
# def load_data():
#     # Load the data (replace with your data loading logic)
#     df = pd.read_csv("uncleaned_traffic.csv")
#
#     # Your data preprocessing steps go here
#     # Data cleaning
#     df = df.dropna(subset=['DateTime', 'Junction', 'Vehicles', 'ID'])
#     df = df.drop_duplicates()
#     df['DateTime'] = pd.to_datetime(df['DateTime'])
#     df['Vehicles'] = pd.to_numeric(df['Vehicles'], errors='coerce')  # Ensure 'Vehicles' column is numeric
#     df = df.dropna(subset=['Vehicles'])
#     df['Junction'] = df['Junction'].astype(int)
#     df['ID'] = df['ID'].apply(lambda x: '{:.0f}'.format(x))
#     df = df.drop_duplicates()
#     df = df.sort_values(by=['Junction', 'DateTime'])
#     df.reset_index(drop=True, inplace=True)
#     df.set_index('DateTime', inplace=True)
#
#     # Handling missing values
#     df.fillna(method='ffill', inplace=True)
#
#     # Extracting date-time features
#     df['Year'] = df.index.year
#     df['Month'] = df.index.month
#     df['Day'] = df.index.day
#     df['Hour'] = df.index.hour
#     df['DayOfWeek'] = df.index.dayofweek
#
#     # Adding cyclical features
#     df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
#     df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
#     df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
#     df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
#
#     # Adding holiday feature
#     cal = calendar()
#     holidays = cal.holidays(start=df.index.min(), end=df.index.max())
#     df['Holiday'] = df.index.isin(holidays).astype(int)
#
#     # Feature scaling
#     scaler = StandardScaler()
#     df['Vehicles_scaled'] = scaler.fit_transform(df[['Vehicles']])
#
#     return df
#
# # Load data
# df = load_data()
#
# # Check if 'Year' column exists in the DataFrame
# if 'Year' not in df.columns:
#     st.error("Year column not found in the DataFrame!")
# else:
#     # Initialize junctions variable
#     junctions = []
#
#
#     # Sidebar filters
#     st.sidebar.title("Traffic Data Filters")
#
#     # Display images of traffic police and department logo
#     st.sidebar.title("Traffic Police")
#     traffic_police_image = Image.open("traffic-police.jpg")
#     st.sidebar.image(traffic_police_image, caption='Traffic Police')
#
#     department_logo_image = Image.open("logo.jpg")
#     st.sidebar.image(department_logo_image, caption='Department Logo')
#
#     # Checkbox for selecting specific date, year, and hour
#     show_date_checkbox = st.sidebar.checkbox("Filter by Date")
#
#     if show_date_checkbox:
#         st.sidebar.title("Date Filters")
#         year = st.sidebar.selectbox("Select Year", df['Year'].unique())
#         month = st.sidebar.selectbox("Select Month", df['Month'].unique())
#         day = st.sidebar.selectbox("Select Day", df['Day'].unique())
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}")
#
#         filtered_df = df[(df['Year'] == year) & (df['Month'] == month) & (df['Day'] == day)]
#         st.write(filtered_df)
#
#     else:
#         st.sidebar.title("Date and Time Filters")
#         year = st.sidebar.selectbox("Select Year", df['Year'].unique())
#         month = st.sidebar.selectbox("Select Month", df['Month'].unique())
#         day = st.sidebar.selectbox("Select Day", df['Day'].unique())
#         hour = st.sidebar.selectbox("Select Hour", df['Hour'].unique())
#         junctions = st.sidebar.multiselect("Select Junction", df['Junction'].unique())
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}, Hour: {hour}, Junctions: {junctions}")
#
#         filtered_df = df[(df['Year'] == year) & (df['Month'] == month) & (df['Day'] == day) & (df['Hour'] == hour) & (
#                     df['Junction'].isin(junctions))]
#         st.write(filtered_df)
#
#     # Add an overall data option
#     show_overall_data = st.sidebar.checkbox("Show Overall Data")
#     if show_overall_data:
#         st.sidebar.write("Showing Overall Data")
#         overall_df = df.groupby(['Year', 'Month', 'Day', 'Hour', 'Junction'], as_index=False)['Vehicles'].mean()
#         st.write(overall_df)
#
#     # Visualization of past data
#     st.title("Historical Traffic Data")
#
#     # Dropdown menu for selecting visualization type
#     visualization_option = st.selectbox("Select Visualization Type", ["Line Chart", "Bar Chart", "Scatter Plot"])
#
#     if visualization_option == "Line Chart":
#         # Dropdown menu for selecting granularity
#         granularity = st.selectbox("Select Granularity", ["Year", "Hour", "Date"])
#
#         if granularity == "Year":
#             if show_overall_data:
#                 historical_df = overall_df.groupby(['Year', 'Junction'], as_index=False)['Vehicles'].mean()
#             else:
#                 historical_df = filtered_df.groupby(['Year', 'Junction'], as_index=False)['Vehicles'].mean()
#
#             fig = px.line(historical_df, x='Year', y='Vehicles', color='Junction',
#                           title='Average Yearly Traffic Data',
#                           labels={'Year': 'Year', 'Vehicles': 'Average Number of Vehicles'})
#             st.plotly_chart(fig)
#
#         elif granularity == "Hour":
#             if show_overall_data:
#                 historical_df = overall_df.groupby(['Hour', 'Junction'], as_index=False)['Vehicles'].mean()
#             else:
#                 historical_df = filtered_df.groupby(['Hour', 'Junction'], as_index=False)['Vehicles'].mean()
#
#             fig = px.line(historical_df, x='Hour', y='Vehicles', color='Junction',
#                           title='Average Hourly Traffic Data',
#                           labels={'Hour': 'Hour of Day', 'Vehicles': 'Average Number of Vehicles'})
#             st.plotly_chart(fig)
#
#         elif granularity == "Date":
#             if show_overall_data:
#                 historical_df = overall_df.groupby(['Year', 'Month', 'Day', 'Junction'], as_index=False)[
#                     'Vehicles'].mean()
#                 historical_df['Date'] = pd.to_datetime(historical_df[['Year', 'Month', 'Day']])
#             else:
#                 historical_df = filtered_df.groupby(['Year', 'Month', 'Day', 'Junction'], as_index=False)[
#                     'Vehicles'].mean()
#                 historical_df['Date'] = pd.to_datetime(historical_df[['Year', 'Month', 'Day']])
#
#             fig = px.line(historical_df, x='Date', y='Vehicles', color='Junction',
#                           title='Average Daily Traffic Data',
#                           labels={'Date': 'Date', 'Vehicles': 'Average Number of Vehicles'})
#             st.plotly_chart(fig)
#
#     elif visualization_option == "Bar Chart":
#         if show_overall_data:
#             fig = px.bar(overall_df, x=overall_df.index, y='Vehicles', color='Junction',
#                          title='Overall Traffic Data',
#                          labels={'DateTime': 'Date and Time', 'Vehicles': 'Number of Vehicles'})
#         else:
#             fig = px.bar(filtered_df, x=filtered_df.index, y='Vehicles', color='Junction',
#                          title='Historical Traffic Data',
#                          labels={'DateTime': 'Date and Time', 'Vehicles': 'Number of Vehicles'})
#         st.plotly_chart(fig)
#
#     elif visualization_option == "Scatter Plot":
#         if show_overall_data:
#             fig = px.scatter(overall_df, x='Hour', y='Vehicles', color='DayOfWeek', symbol='Junction',
#                              title='Overall Scatter Plot of Traffic Data',
#                              labels={'Hour': 'Hour of Day', 'Vehicles': 'Number of Vehicles',
#                                      'DayOfWeek': 'Day of Week'})
#         else:
#             fig = px.scatter(filtered_df, x='Hour', y='Vehicles', color='DayOfWeek', symbol='Junction',
#                              title='Scatter Plot of Traffic Data',
#                              labels={'Hour': 'Hour of Day', 'Vehicles': 'Number of Vehicles',
#                                      'DayOfWeek': 'Day of Week'})
#         st.plotly_chart(fig)
#
#     # Allow users to update traffic info
#     st.sidebar.title("Update Traffic Information")
#     new_traffic_info = st.sidebar.number_input("Enter new traffic count", min_value=0, value=0)
#     if st.sidebar.button("Update Traffic Count"):
#         # Update logic here
#         if not show_date_checkbox:
#             st.write(
#                 f"New traffic count of {new_traffic_info} at Junctions {junctions} on {year}-{month}-{day} at {hour}:00 updated successfully.")
#         else:
#             st.write(
#                 f"New traffic count of {new_traffic_info} on {year}-{month}-{day} updated successfully.")
#
#     # Allow users to upload photos
#     st.sidebar.title("Upload Photos")
#     uploaded_files = st.sidebar.file_uploader("Choose a photo", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
#     for uploaded_file in uploaded_files:
#         st.sidebar.image(uploaded_file, caption=uploaded_file.name)
#
# import streamlit as st
# import pandas as pd
# import numpy as np
# from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
# from sklearn.preprocessing import StandardScaler
# import plotly.express as px
# from PIL import Image
#
# # Load and preprocess data
# @st.cache(allow_output_mutation=True)
# def load_data():
#     # Load the data (replace with your data loading logic)
#     df = pd.read_csv("uncleaned_traffic.csv")
#
#     # Your data preprocessing steps go here
#     # Data cleaning
#     df = df.dropna(subset=['DateTime', 'Junction', 'Vehicles', 'ID'])
#     df = df.drop_duplicates()
#     df['DateTime'] = pd.to_datetime(df['DateTime'])
#     df['Vehicles'] = pd.to_numeric(df['Vehicles'], errors='coerce')  # Ensure 'Vehicles' column is numeric
#     df = df.dropna(subset=['Vehicles'])
#     df['Junction'] = df['Junction'].astype(int)
#     df['ID'] = df['ID'].apply(lambda x: '{:.0f}'.format(x))
#     df = df.drop_duplicates()
#     df = df.sort_values(by=['Junction', 'DateTime'])
#     df.reset_index(drop=True, inplace=True)
#     df.set_index('DateTime', inplace=True)
#
#     # Handling missing values
#     df.fillna(method='ffill', inplace=True)
#
#     # Extracting date-time features
#     df['Year'] = df.index.year
#     df['Month'] = df.index.month
#     df['Day'] = df.index.day
#     df['Hour'] = df.index.hour
#     df['DayOfWeek'] = df.index.dayofweek
#
#     # Adding cyclical features
#     df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
#     df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
#     df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
#     df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
#
#     # Adding holiday feature
#     cal = calendar()
#     holidays = cal.holidays(start=df.index.min(), end=df.index.max())
#     df['Holiday'] = df.index.isin(holidays).astype(int)
#
#     # Feature scaling
#     scaler = StandardScaler()
#     df['Vehicles_scaled'] = scaler.fit_transform(df[['Vehicles']])
#
#     return df
#
# # Load data
# df = load_data()
#
# # Check if 'Year' column exists in the DataFrame
# if 'Year' not in df.columns:
#     st.error("Year column not found in the DataFrame!")
# else:
#     # Initialize junctions variable
#     junctions = []
#
#     # Sidebar filters
#     st.sidebar.title("Traffic Data Filters")
#
#     # Display images of traffic police and department logo
#     st.sidebar.title("Traffic Police")
#     traffic_police_image = Image.open("traffic-police.jpg")
#     st.sidebar.image(traffic_police_image, caption='Traffic Police')
#
#     department_logo_image = Image.open("logo.jpg")
#     st.sidebar.image(department_logo_image, caption='Department Logo')
#
#     # Checkbox for selecting specific date, year, and hour
#     show_date_checkbox = st.sidebar.checkbox("Filter by Date")
#
#     if show_date_checkbox:
#         st.sidebar.title("Date Filters")
#         year = st.sidebar.selectbox("Select Year", df['Year'].unique())
#         month = st.sidebar.selectbox("Select Month", df['Month'].unique())
#         day = st.sidebar.selectbox("Select Day", df['Day'].unique())
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}")
#
#         filtered_df = df[(df['Year'] == year) & (df['Month'] == month) & (df['Day'] == day)]
#         st.write(filtered_df)
#
#         # Visualization of filtered data
#         st.title("Historical Traffic Data - Line Chart")
#
#         # Group by Hour and Junction to get the average number of vehicles
#         historical_df = filtered_df.groupby(['Hour', 'Junction'], as_index=False)['Vehicles'].mean()
#
#         # Create line chart using Plotly Express
#         fig = px.line(historical_df, x='Hour', y='Vehicles', color='Junction',
#                       title=f'Average Hourly Traffic Data for {month}/{day}/{year}',
#                       labels={'Hour': 'Hour of Day', 'Vehicles': 'Average Number of Vehicles'})
#         st.plotly_chart(fig)
#
#     else:
#         st.sidebar.title("Date and Time Filters")
#         year = st.sidebar.selectbox("Select Year", df['Year'].unique())
#         month = st.sidebar.selectbox("Select Month", df['Month'].unique())
#         day = st.sidebar.selectbox("Select Day", df['Day'].unique())
#         hour = st.sidebar.selectbox("Select Hour", df['Hour'].unique())
#         junctions = st.sidebar.multiselect("Select Junction", df['Junction'].unique())
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}, Hour: {hour}, Junctions: {junctions}")
#
#         filtered_df = df[(df['Year'] == year) & (df['Month'] == month) & (df['Day'] == day) & (df['Hour'] == hour) & (
#                     df['Junction'].isin(junctions))]
#         st.write(filtered_df)
#
#         # Add an overall data option
#         show_overall_data = st.sidebar.checkbox("Show Overall Data")
#         if show_overall_data:
#             st.sidebar.write("Showing Overall Data")
#             overall_df = df.groupby(['Year', 'Month', 'Day', 'Hour', 'Junction'], as_index=False)['Vehicles'].mean()
#             st.write(overall_df)
#
#         # Visualization of past data
#         st.title("Historical Traffic Data")
#
#         # Dropdown menu for selecting visualization type
#         visualization_option = st.selectbox("Select Visualization Type", ["Line Chart", "Bar Chart", "Scatter Plot"])
#
#         if visualization_option == "Line Chart":
#             # Dropdown menu for selecting granularity
#             granularity = st.selectbox("Select Granularity", ["Year", "Hour", "Date"])
#
#             if granularity == "Year":
#                 if show_overall_data:
#                     historical_df = overall_df.groupby(['Year', 'Junction'], as_index=False)['Vehicles'].mean()
#                 else:
#                     historical_df = filtered_df.groupby(['Year', 'Junction'], as_index=False)['Vehicles'].mean()
#
#                 fig = px.line(historical_df, x='Year', y='Vehicles', color='Junction',
#                               title='Average Yearly Traffic Data',
#                               labels={'Year': 'Year', 'Vehicles': 'Average Number of Vehicles'})
#                 st.plotly_chart(fig)
#
#             elif granularity == "Hour":
#                 if show_overall_data:
#                     historical_df = overall_df.groupby(['Hour', 'Junction'], as_index=False)['Vehicles'].mean()
#                 else:
#                     historical_df = filtered_df.groupby(['Hour', 'Junction'], as_index=False)['Vehicles'].mean()
#
#                 fig = px.line(historical_df, x='Hour', y='Vehicles', color='Junction',
#                               title='Average Hourly Traffic Data',
#                               labels={'Hour': 'Hour of Day', 'Vehicles': 'Average Number of Vehicles'})
#                 st.plotly_chart(fig)
#
#             elif granularity == "Date":
#                 if show_overall_data:
#                     historical_df = overall_df.groupby(['Year', 'Month', 'Day', 'Junction'], as_index=False)[
#                         'Vehicles'].mean()
#                     historical_df['Date'] = pd.to_datetime(historical_df[['Year', 'Month', 'Day']])
#                 else:
#                     historical_df = filtered_df.groupby(['Year', 'Month', 'Day', 'Junction'], as_index=False)[
#                         'Vehicles'].mean()
#                     historical_df['Date'] = pd.to_datetime(historical_df[['Year', 'Month', 'Day']])
#
#                 fig = px.line(historical_df, x='Date', y='Vehicles', color='Junction',
#                               title='Average Daily Traffic Data',
#                               labels={'Date': 'Date', 'Vehicles': 'Average Number of Vehicles'})
#                 st.plotly_chart(fig)
#
#         elif visualization_option == "Bar Chart":
#             if show_overall_data:
#                 fig = px.bar(overall_df, x=overall_df.index, y='Vehicles', color='Junction',
#                              title='Overall Traffic Data',
#                              labels={'DateTime': 'Date and Time', 'Vehicles': 'Number of Vehicles'})
#             else:
#                 fig = px.bar(filtered_df, x=filtered_df.index, y='Vehicles', color='Junction',
#                              title='Historical Traffic Data',
#                              labels={'DateTime': 'Date and Time', 'Vehicles': 'Number of Vehicles'})
#             st.plotly_chart(fig)
#
#         elif visualization_option == "Scatter Plot":
#             if show_overall_data:
#                 fig = px.scatter(overall_df, x='Hour', y='Vehicles', color='DayOfWeek', symbol='Junction',
#                                  title='Overall Scatter Plot of Traffic Data',
#                                  labels={'Hour': 'Hour of Day', 'Vehicles': 'Number of Vehicles',
#                                          'DayOfWeek': 'Day of Week'})
#             else:
#                 fig = px.scatter(filtered_df, x='Hour', y='Vehicles', color='DayOfWeek', symbol='Junction',
#                                  title='Scatter Plot of Traffic Data',
#                                  labels={'Hour': 'Hour of Day', 'Vehicles': 'Number of Vehicles',
#                                          'DayOfWeek': 'Day of Week'})
#                 st.plotly_chart(fig)
#
#             st.plotly_chart(fig)
#
#     # Allow users to update traffic info
#     st.sidebar.title("Update Traffic Information")
#     new_traffic_info = st.sidebar.number_input("Enter new traffic count", min_value=0, value=0)
#     if st.sidebar.button("Update Traffic Count"):
#         # Update logic here
#         if not show_date_checkbox:
#             st.write(
#                 f"New traffic count of {new_traffic_info} at Junctions {junctions} on {year}-{month}-{day} at {hour}:00 updated successfully.")
#         else:
#             st.write(
#                 f"New traffic count of {new_traffic_info} on {year}-{month}-{day} updated successfully.")
#
#     # Allow users to upload photos
#     st.sidebar.title("Upload Photos")
#     uploaded_files = st.sidebar.file_uploader("Choose a photo", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
#     for uploaded_file in uploaded_files:
#         st.sidebar.image(uploaded_file, caption=uploaded_file.name)

# import streamlit as st
# import pandas as pd
# import numpy as np
# from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
# from sklearn.preprocessing import StandardScaler
# import plotly.express as px
# from PIL import Image
#
#
# # Load and preprocess data
# @st.cache(allow_output_mutation=True)
# def load_data():
#     # Load the data (replace with your data loading logic)
#     df = pd.read_csv("uncleaned_traffic.csv")
#
#     # Your data preprocessing steps go here
#     # Data cleaning
#     df = df.dropna(subset=['DateTime', 'Junction', 'Vehicles', 'ID'])
#     df = df.drop_duplicates()
#     df['DateTime'] = pd.to_datetime(df['DateTime'])
#     df['Vehicles'] = pd.to_numeric(df['Vehicles'], errors='coerce')  # Ensure 'Vehicles' column is numeric
#     df = df.dropna(subset=['Vehicles'])
#     df['Junction'] = df['Junction'].astype(int)
#     df['ID'] = df['ID'].apply(lambda x: '{:.0f}'.format(x))
#     df = df.drop_duplicates()
#     df = df.sort_values(by=['Junction', 'DateTime'])
#     df.reset_index(drop=True, inplace=True)
#     df.set_index('DateTime', inplace=True)
#
#     # Handling missing values
#     df.fillna(method='ffill', inplace=True)
#
#     # Extracting date-time features
#     df['Year'] = df.index.year
#     df['Month'] = df.index.month
#     df['Day'] = df.index.day
#     df['Hour'] = df.index.hour
#     df['DayOfWeek'] = df.index.dayofweek
#
#     # Adding cyclical features
#     df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
#     df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
#     df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
#     df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
#
#     # Adding holiday feature
#     cal = calendar()
#     holidays = cal.holidays(start=df.index.min(), end=df.index.max())
#     df['Holiday'] = df.index.isin(holidays).astype(int)
#
#     # Feature scaling
#     scaler = StandardScaler()
#     df['Vehicles_scaled'] = scaler.fit_transform(df[['Vehicles']])
#
#     return df
#
#
# # Load data
# df = load_data()
#
# # Check if 'Year' column exists in the DataFrame
# if 'Year' not in df.columns:
#     st.error("Year column not found in the DataFrame!")
# else:
#     # Initialize junctions variable
#     junctions = []
#
#     # Sidebar filters
#     st.sidebar.title("Traffic Data Filters")
#
#     # Display images of traffic police and department logo
#     st.sidebar.title("Traffic Police")
#     traffic_police_image = Image.open("traffic-police.jpg")
#     st.sidebar.image(traffic_police_image, caption='Traffic Police')
#
#     department_logo_image = Image.open("logo.jpg")
#     st.sidebar.image(department_logo_image, caption='Department Logo')
#
#     # Dropdown select boxes for filtering
#     st.sidebar.title("Date and Time Filters")
#     year = st.sidebar.selectbox("Select Year", df['Year'].unique())
#     month = st.sidebar.selectbox("Select Month", df['Month'].unique())
#     day = st.sidebar.selectbox("Select Day", df['Day'].unique())
#     hour = st.sidebar.selectbox("Select Hour", df['Hour'].unique())
#
#     st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}, Hour: {hour}")
#
#     # Filter data based on user selection
#     filtered_df = df[(df['Year'] == year) & (df['Month'] == month) & (df['Day'] == day) & (df['Hour'] == hour)]
#
#     # Display filtered data
#     st.write(filtered_df)
#
#     # Visualization of filtered data
#     st.title("Historical Traffic Data - Line Chart")
#
#     # Group by Hour and Junction to get the average number of vehicles
#     historical_df = filtered_df.groupby(['Hour', 'Junction'], as_index=False)['Vehicles'].mean()
#
#     # Create line chart using Plotly Express
#     fig = px.line(historical_df, x='Hour', y='Vehicles', color='Junction',
#                   title=f'Average Hourly Traffic Data for {month}/{day}/{year} at Hour {hour}',
#                   labels={'Hour': 'Hour of Day', 'Vehicles': 'Average Number of Vehicles'})
#     st.plotly_chart(fig)
#
#     # Additional data display (if needed)
#     # Example: Displaying a summary or statistics
#     st.title("Additional Data Display")
#     st.write(f"Total number of records: {len(filtered_df)}")
#
#     # Allow users to update traffic info
#     st.sidebar.title("Update Traffic Information")
#     new_traffic_info = st.sidebar.number_input("Enter new traffic count", min_value=0, value=0)
#     if st.sidebar.button("Update Traffic Count"):
#         # Update logic here
#         st.write(
#             f"New traffic count of {new_traffic_info} at {hour}:00 on {month}/{day}/{year} updated successfully.")
#
#     # Allow users to upload photos
#     st.sidebar.title("Upload Photos")
#     uploaded_files = st.sidebar.file_uploader("Choose a photo", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
#     for uploaded_file in uploaded_files:
#         st.sidebar.image(uploaded_file, caption=uploaded_file.name)
# import base64
#
# import streamlit as st
# import pandas as pd
# import numpy as np
# from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
# from sklearn.preprocessing import StandardScaler
# import plotly.express as px
# from PIL import Image
#
# # Load and preprocess data
# @st.cache(allow_output_mutation=True)
# def load_data():
#     # Load the data (replace with your data loading logic)
#     df = pd.read_csv("uncleaned_traffic.csv")
#
#     # Your data preprocessing steps go here
#     # Data cleaning
#     df = df.dropna(subset=['DateTime', 'Junction', 'Vehicles', 'ID'])
#     df = df.drop_duplicates()
#     df['DateTime'] = pd.to_datetime(df['DateTime'])
#     df['Vehicles'] = pd.to_numeric(df['Vehicles'], errors='coerce')  # Ensure 'Vehicles' column is numeric
#     df = df.dropna(subset=['Vehicles'])
#     df['Junction'] = df['Junction'].astype(int)
#     df['ID'] = df['ID'].apply(lambda x: '{:.0f}'.format(x))
#     df = df.drop_duplicates()
#     df = df.sort_values(by=['Junction', 'DateTime'])
#     df.reset_index(drop=True, inplace=True)
#     df.set_index('DateTime', inplace=True)
#
#     # Handling missing values
#     df.fillna(method='ffill', inplace=True)
#
#     # Extracting date-time features
#     df['Year'] = df.index.year
#     df['Month'] = df.index.month
#     df['Day'] = df.index.day
#     df['Hour'] = df.index.hour
#     df['DayOfWeek'] = df.index.dayofweek
#
#     # Adding cyclical features
#     df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
#     df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
#     df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
#     df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
#
#     # Adding holiday feature
#     cal = calendar()
#     holidays = cal.holidays(start=df.index.min(), end=df.index.max())
#     df['Holiday'] = df.index.isin(holidays).astype(int)
#
#     # Feature scaling
#     scaler = StandardScaler()
#     df['Vehicles_scaled'] = scaler.fit_transform(df[['Vehicles']])
#
#     return df
#
# # Load data
# df = load_data()
#
# # Check if 'Year' column exists in the DataFrame
# if 'Year' not in df.columns:
#     st.error("Year column not found in the DataFrame!")
# else:
#     # Initialize junctions variable
#     junctions = []
#
#     # Sidebar filters
#     st.sidebar.title("Traffic Data Filters")
#
#     # Display images of traffic police and department logo
#     st.sidebar.title("Traffic Police")
#     traffic_police_image = Image.open("traffic-police.jpg")
#     st.sidebar.image(traffic_police_image, caption='Traffic Police')
#
#     department_logo_image = Image.open("logo.jpg")
#     st.sidebar.image(department_logo_image, caption='Department Logo')
#
#     # Checkbox for selecting specific date, year, and hour
#     show_date_checkbox = st.sidebar.checkbox("Filter by Date")
#
#     if show_date_checkbox:
#         st.sidebar.title("Date Filters")
#         year = st.sidebar.selectbox("Select Year", df['Year'].unique())
#         month = st.sidebar.selectbox("Select Month", df['Month'].unique())
#         day = st.sidebar.selectbox("Select Day", df['Day'].unique())
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}")
#
#         filtered_df = df[(df['Year'] == year) & (df['Month'] == month) & (df['Day'] == day)]
#     else:
#         st.sidebar.title("Date and Time Filters")
#         year = st.sidebar.selectbox("Select Year", df['Year'].unique())
#         month = st.sidebar.selectbox("Select Month", df['Month'].unique())
#         day = st.sidebar.selectbox("Select Day", df['Day'].unique())
#         hour = st.sidebar.selectbox("Select Hour", df['Hour'].unique())
#         junctions = st.sidebar.multiselect("Select Junction", df['Junction'].unique())
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}, Hour: {hour}, Junctions: {junctions}")
#
#         filtered_df = df[(df['Year'] == year) & (df['Month'] == month) & (df['Day'] == day) & (df['Hour'] == hour) & (
#                     df['Junction'].isin(junctions))]
#
#     # Display filtered data
#     st.write(filtered_df)
#
#     # Visualization of filtered data
#     st.title("Historical Traffic Data - Line Chart")
#
#     if not filtered_df.empty:
#         # Group by Hour and Junction to get the average number of vehicles
#         historical_df = filtered_df.groupby(['Hour', 'Junction'], as_index=False)['Vehicles'].mean()
#
#         # Create line chart using Plotly Express
#         fig = px.line(historical_df, x='Hour', y='Vehicles', color='Junction',
#                       title='Average Hourly Traffic Data',
#                       labels={'Hour': 'Hour of Day', 'Vehicles': 'Average Number of Vehicles'})
#         st.plotly_chart(fig)
#
#     # Additional data display (if needed)
#     # Example: Displaying a summary or statistics
#     st.title("Additional Data Display")
#     if not filtered_df.empty:
#         st.write(f"Total number of records: {len(filtered_df)}")
#
#     # Allow users to update traffic info
#     st.sidebar.title("Update Traffic Information")
#     new_traffic_info = st.sidebar.number_input("Enter new traffic count", min_value=0, value=0)
#     if st.sidebar.button("Update Traffic Count"):
#         # Update logic here
#         if show_date_checkbox:
#             st.write(
#                 f"New traffic count of {new_traffic_info} on {year}-{month}-{day} updated successfully.")
#         else:
#             st.write(
#                 f"New traffic count of {new_traffic_info} at {hour}:00 on {month}/{day}/{year} at Junctions {junctions} updated successfully.")
#
#     # Allow users to upload photos
#     st.sidebar.title("Upload Photos")
#     uploaded_files = st.sidebar.file_uploader("Choose a photo", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
#     for uploaded_file in uploaded_files:
#         st.sidebar.image(uploaded_file, caption=uploaded_file.name)
#
#         # Display submitted photos
#         for uploaded_file in uploaded_files:
#             st.sidebar.image(uploaded_file, caption=uploaded_file.name)
#
#         # Submit button to process uploaded photos and generate downloadable file
#         if st.sidebar.button("Submit Photos"):
#             # Example: Save uploaded files to a PDF or CSV (here we save to CSV for simplicity)
#             photos_data = []
#
#             for uploaded_file in uploaded_files:
#                 # Convert uploaded image to base64
#                 encoded_img = base64.b64encode(uploaded_file.read()).decode('utf-8')
#                 photos_data.append({'Filename': uploaded_file.name, 'Image': encoded_img})
#
#             # Create a DataFrame from photos_data
#             photos_df = pd.DataFrame(photos_data)
#
#             # Download link
#             csv = photos_df.to_csv(index=False)
#             b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
#             href = f'<a href="data:file/csv;base64,{b64}" download="uploaded_photos.csv">Download photos data as CSV file</a>'
#             st.sidebar.markdown(href, unsafe_allow_html=True)
# import streamlit as st
# import pandas as pd
# import numpy as np
# from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
# from sklearn.preprocessing import StandardScaler
# import plotly.express as px
# from PIL import Image
# import base64
# import io
# import os
#
# # Global variables
# uploaded_files = []
#
# # Load and preprocess data
# @st.cache(allow_output_mutation=True)
# def load_data():
#     # Load the data (replace with your data loading logic)
#     df = pd.read_csv("uncleaned_traffic.csv")
#
#     # Your data preprocessing steps go here
#     # Data cleaning
#     df = df.dropna(subset=['DateTime', 'Junction', 'Vehicles', 'ID'])
#     df = df.drop_duplicates()
#     df['DateTime'] = pd.to_datetime(df['DateTime'])
#     df['Vehicles'] = pd.to_numeric(df['Vehicles'], errors='coerce')  # Ensure 'Vehicles' column is numeric
#     df = df.dropna(subset=['Vehicles'])
#     df['Junction'] = df['Junction'].astype(int)
#     df['ID'] = df['ID'].apply(lambda x: '{:.0f}'.format(x))
#     df = df.drop_duplicates()
#     df = df.sort_values(by=['Junction', 'DateTime'])
#     df.reset_index(drop=True, inplace=True)
#     df.set_index('DateTime', inplace=True)
#
#     # Handling missing values
#     df.fillna(method='ffill', inplace=True)
#
#     # Extracting date-time features
#     df['Year'] = df.index.year
#     df['Month'] = df.index.month
#     df['Day'] = df.index.day
#     df['Hour'] = df.index.hour
#     df['DayOfWeek'] = df.index.dayofweek
#
#     # Adding cyclical features
#     df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
#     df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
#     df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
#     df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
#
#     # Adding holiday feature
#     cal = calendar()
#     holidays = cal.holidays(start=df.index.min(), end=df.index.max())
#     df['Holiday'] = df.index.isin(holidays).astype(int)
#
#     # Feature scaling
#     scaler = StandardScaler()
#     df['Vehicles_scaled'] = scaler.fit_transform(df[['Vehicles']])
#
#     return df
#
# # Load data
# df = load_data()
#
# # Check if 'Year' column exists in the DataFrame
# if 'Year' not in df.columns:
#     st.error("Year column not found in the DataFrame!")
# else:
#     # Initialize junctions variable
#     junctions = []
#
#     # Sidebar filters
#     st.sidebar.title("Traffic Data Filters")
#
#     # Display images of traffic police and department logo
#     st.sidebar.title("Traffic Police")
#     traffic_police_image = Image.open("traffic-police.jpg")
#     st.sidebar.image(traffic_police_image, caption='Traffic Police')
#
#     department_logo_image = Image.open("logo.jpg")
#     st.sidebar.image(department_logo_image, caption='Department Logo')
#
#     # Checkbox for selecting specific date, year, and hour
#     show_date_checkbox = st.sidebar.checkbox("Filter by Date")
#
#     if show_date_checkbox:
#         st.sidebar.title("Date Filters")
#         year = st.sidebar.selectbox("Select Year", df['Year'].unique())
#         month = st.sidebar.selectbox("Select Month", df['Month'].unique())
#         day = st.sidebar.selectbox("Select Day", df['Day'].unique())
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}")
#
#         filtered_df = df[(df['Year'] == year) & (df['Month'] == month) & (df['Day'] == day)]
#     else:
#         st.sidebar.title("Date and Time Filters")
#         year = st.sidebar.selectbox("Select Year", df['Year'].unique())
#         month = st.sidebar.selectbox("Select Month", df['Month'].unique())
#         day = st.sidebar.selectbox("Select Day", df['Day'].unique())
#         hour = st.sidebar.selectbox("Select Hour", df['Hour'].unique())
#         junctions = st.sidebar.multiselect("Select Junction", df['Junction'].unique())
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}, Hour: {hour}, Junctions: {junctions}")
#
#         filtered_df = df[(df['Year'] == year) & (df['Month'] == month) & (df['Day'] == day) & (df['Hour'] == hour) & (
#                     df['Junction'].isin(junctions))]
#
#     # Display filtered data
#     st.write(filtered_df)
#
#     # Visualization of filtered data
#     st.title("Historical Traffic Data - Line Chart")
#
#     if not filtered_df.empty:
#         # Group by Hour and Junction to get the average number of vehicles
#         historical_df = filtered_df.groupby(['Hour', 'Junction'], as_index=False)['Vehicles'].mean()
#
#         # Create line chart using Plotly Express
#         fig = px.line(historical_df, x='Hour', y='Vehicles', color='Junction',
#                       title='Average Hourly Traffic Data',
#                       labels={'Hour': 'Hour of Day', 'Vehicles': 'Average Number of Vehicles'})
#         st.plotly_chart(fig)
#
#     # Additional data display (if needed)
#     # Example: Displaying a summary or statistics
#     st.title("Additional Data Display")
#     if not filtered_df.empty:
#         st.write(f"Total number of records: {len(filtered_df)}")
#
#     # Allow users to update traffic info
#     st.sidebar.title("Update Traffic Information")
#     new_traffic_info = st.sidebar.number_input("Enter new traffic count", min_value=0, value=0)
#     if st.sidebar.button("Update Traffic Count"):
#         # Update logic here
#         if show_date_checkbox:
#             st.write(
#                 f"New traffic count of {new_traffic_info} on {year}-{month}-{day} updated successfully.")
#         else:
#             st.write(
#                 f"New traffic count of {new_traffic_info} at {hour}:00 on {month}/{day}/{year} at Junctions {junctions} updated successfully.")
#
#     # Allow users to upload, update, and delete photos
#     st.sidebar.title("Manage Photos")
#
#     if st.sidebar.button("Upload Photo"):
#         uploaded_file = st.sidebar.file_uploader("Choose a photo", type=["jpg", "jpeg", "png"])
#         if uploaded_file is not None:
#             uploaded_files.append(uploaded_file)
#
#     # Display submitted photos
#     if uploaded_files:
#         st.sidebar.title("Submitted Photos")
#         for uploaded_file in uploaded_files:
#             st.sidebar.image(uploaded_file, caption=uploaded_file.name)
#
#     # Show only one image
#     if uploaded_files:
#         st.sidebar.image(uploaded_files[0], caption=uploaded_files[0].name)
#
#     # Option to delete photo
#     photo_to_delete = st.sidebar.selectbox("Select photo to delete", [uploaded_file.name for uploaded_file in uploaded_files])
#     if st.sidebar.button("Delete Photo"):
#         uploaded_files = [file for file in uploaded_files if file.name != photo_to_delete]
#
#     # Submit button to process uploaded photos and generate downloadable file
#     if st.sidebar.button("Submit Photos"):
#         # Example: Save uploaded files to a CSV with links
#         photos_data = [{'Filename': file.name, 'Link': file.name} for file in uploaded_files]
#
#         # Create a DataFrame from photos_data
#         photos_df = pd.DataFrame(photos_data)
#
#         # Save photos_df to CSV
#         photos_csv = photos_df.to_csv(index=False)
#
#         # Download link for CSV file
#         csv_download_link = f'<a href="data:file/csv;base64,{base64.b64encode(photos_csv.encode()).decode()}" download="uploaded_photos.csv">Download photos data as CSV file</a>'
#         st.sidebar.markdown(csv_download_link, unsafe_allow_html=True)

# import streamlit as st
# import pandas as pd
# import numpy as np
# from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
# from sklearn.preprocessing import StandardScaler
# import plotly.express as px
# from PIL import Image
#
# # Load and preprocess data
# @st.cache(allow_output_mutation=True)
# def load_data():
#     # Load the data (replace with your data loading logic)
#     df = pd.read_csv("uncleaned_traffic.csv")
#
#     # Your data preprocessing steps go here
#     # Data cleaning
#     df = df.dropna(subset=['DateTime', 'Junction', 'Vehicles', 'ID'])
#     df = df.drop_duplicates()
#     df['DateTime'] = pd.to_datetime(df['DateTime'])
#     df['Vehicles'] = pd.to_numeric(df['Vehicles'], errors='coerce')  # Ensure 'Vehicles' column is numeric
#     df = df.dropna(subset=['Vehicles'])
#     df['Junction'] = df['Junction'].astype(int)
#     df['ID'] = df['ID'].apply(lambda x: '{:.0f}'.format(x))
#     df = df.drop_duplicates()
#     df = df.sort_values(by=['Junction', 'DateTime'])
#     df.reset_index(drop=True, inplace=True)
#     df.set_index('DateTime', inplace=True)
#
#     # Handling missing values
#     df.fillna(method='ffill', inplace=True)
#
#     # Extracting date-time features
#     df['Year'] = df.index.year
#     df['Month'] = df.index.month
#     df['Day'] = df.index.day
#     df['Hour'] = df.index.hour
#     df['DayOfWeek'] = df.index.dayofweek
#
#     # Adding cyclical features
#     df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
#     df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
#     df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
#     df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
#
#     # Adding holiday feature
#     cal = calendar()
#     holidays = cal.holidays(start=df.index.min(), end=df.index.max())
#     df['Holiday'] = df.index.isin(holidays).astype(int)
#
#     # Feature scaling
#     scaler = StandardScaler()
#     df['Vehicles_scaled'] = scaler.fit_transform(df[['Vehicles']])
#
#     return df
#
# # Load data
# df = load_data()
#
# # Check if 'Year' column exists in the DataFrame
# if 'Year' not in df.columns:
#     st.error("Year column not found in the DataFrame!")
# else:
#     # Initialize junctions variable
#     junctions = []
#
#     # Sidebar filters
#     st.sidebar.title("Traffic Data Filters")
#
#     # Display images of traffic police and department logo
#     st.sidebar.title("Traffic Police")
#     traffic_police_image = Image.open("traffic-police.jpg")
#     st.sidebar.image(traffic_police_image, caption='Traffic Police')
#
#     department_logo_image = Image.open("logo.jpg")
#     st.sidebar.image(department_logo_image, caption='Department Logo')
#
#     # Checkbox for selecting specific date, year, and hour
#     show_date_checkbox = st.sidebar.checkbox("Filter by Date")
#
#     if show_date_checkbox:
#         st.sidebar.title("Date Filters")
#         year = st.sidebar.selectbox("Select Year", df['Year'].unique())
#         month = st.sidebar.selectbox("Select Month", df['Month'].unique())
#         day = st.sidebar.selectbox("Select Day", df['Day'].unique())
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}")
#
#         filtered_df = df[(df['Year'] == year) & (df['Month'] == month) & (df['Day'] == day)]
#         st.write(filtered_df)
#
#         # Visualization of filtered data
#         st.title("Historical Traffic Data - Line Chart")
#
#         # Group by Hour and Junction to get the average number of vehicles
#         historical_df = filtered_df.groupby(['Hour', 'Junction'], as_index=False)['Vehicles'].mean()
#
#         # Create line chart using Plotly Express
#         fig = px.line(historical_df, x='Hour', y='Vehicles', color='Junction',
#                       title=f'Average Hourly Traffic Data for {month}/{day}/{year}',
#                       labels={'Hour': 'Hour of Day', 'Vehicles': 'Average Number of Vehicles'})
#         st.plotly_chart(fig)
#
#     else:
#         st.sidebar.title("Date and Time Filters")
#         year = st.sidebar.selectbox("Select Year", df['Year'].unique())
#         month = st.sidebar.selectbox("Select Month", df['Month'].unique())
#         day = st.sidebar.selectbox("Select Day", df['Day'].unique())
#         hour = st.sidebar.selectbox("Select Hour", df['Hour'].unique())
#         junctions = st.sidebar.multiselect("Select Junction", df['Junction'].unique())
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}, Hour: {hour}, Junctions: {junctions}")
#
#         filtered_df = df[(df['Year'] == year) & (df['Month'] == month) & (df['Day'] == day) & (df['Hour'] == hour) & (
#                     df['Junction'].isin(junctions))]
#         st.write(filtered_df)
#
#         # Add an overall data option
#         show_overall_data = st.sidebar.checkbox("Show Overall Data")
#         if show_overall_data:
#             st.sidebar.write("Showing Overall Data")
#             overall_df = df.groupby(['Year', 'Month', 'Day', 'Hour', 'Junction'], as_index=False)['Vehicles'].mean()
#             st.write(overall_df)
#
#         # Visualization of past data
#         st.title("Historical Traffic Data")
#
#         # Dropdown menu for selecting visualization type
#         visualization_option = st.selectbox("Select Visualization Type", ["Line Chart", "Bar Chart", "Scatter Plot"])
#
#         if visualization_option == "Line Chart":
#             # Dropdown menu for selecting granularity
#             granularity = st.selectbox("Select Granularity", ["Year", "Hour", "Date"])
#
#             if granularity == "Year":
#                 if show_overall_data:
#                     historical_df = overall_df.groupby(['Year', 'Junction'], as_index=False)['Vehicles'].mean()
#                 else:
#                     historical_df = filtered_df.groupby(['Year', 'Junction'], as_index=False)['Vehicles'].mean()
#
#                 fig = px.line(historical_df, x='Year', y='Vehicles', color='Junction',
#                               title='Average Yearly Traffic Data',
#                               labels={'Year': 'Year', 'Vehicles': 'Average Number of Vehicles'})
#                 st.plotly_chart(fig)
#
#             elif granularity == "Hour":
#                 if show_overall_data:
#                     historical_df = overall_df.groupby(['Hour', 'Junction'], as_index=False)['Vehicles'].mean()
#                 else:
#                     historical_df = filtered_df.groupby(['Hour', 'Junction'], as_index=False)['Vehicles'].mean()
#
#                 fig = px.line(historical_df, x='Hour', y='Vehicles', color='Junction',
#                               title='Average Hourly Traffic Data',
#                               labels={'Hour': 'Hour of Day', 'Vehicles': 'Average Number of Vehicles'})
#                 st.plotly_chart(fig)
#
#             elif granularity == "Date":
#                 if show_overall_data:
#                     historical_df = overall_df.groupby(['Year', 'Month', 'Day', 'Junction'], as_index=False)[
#                         'Vehicles'].mean()
#                     historical_df['Date'] = pd.to_datetime(historical_df[['Year', 'Month', 'Day']])
#                 else:
#                     historical_df = filtered_df.groupby(['Year', 'Month', 'Day', 'Junction'], as_index=False)[
#                         'Vehicles'].mean()
#                     historical_df['Date'] = pd.to_datetime(historical_df[['Year', 'Month', 'Day']])
#
#                 fig = px.line(historical_df, x='Date', y='Vehicles', color='Junction',
#                               title='Average Daily Traffic Data',
#                               labels={'Date': 'Date', 'Vehicles': 'Average Number of Vehicles'})
#                 st.plotly_chart(fig)
#
#         elif visualization_option == "Bar Chart":
#             if show_overall_data:
#                 fig = px.bar(overall_df, x=overall_df.index, y='Vehicles', color='Junction',
#                              title='Overall Traffic Data',
#                              labels={'DateTime': 'Date and Time', 'Vehicles': 'Number of Vehicles'})
#             else:
#                 fig = px.bar(filtered_df, x=filtered_df.index, y='Vehicles', color='Junction',
#                              title='Historical Traffic Data',
#                              labels={'DateTime': 'Date and Time', 'Vehicles': 'Number of Vehicles'})
#             st.plotly_chart(fig)
#
#         elif visualization_option == "Scatter Plot":
#             if show_overall_data:
#                 fig = px.scatter(overall_df, x='Hour', y='Vehicles', color='DayOfWeek', symbol='Junction',
#                                  title='Overall Scatter Plot of Traffic Data',
#                                  labels={'Hour': 'Hour of Day', 'Vehicles': 'Number of Vehicles',
#                                          'DayOfWeek': 'Day of Week'})
#             else:
#                 fig = px.scatter(filtered_df, x='Hour', y='Vehicles', color='DayOfWeek', symbol='Junction',
#                                  title='Scatter Plot of Traffic Data',
#                                  labels={'Hour': 'Hour of Day', 'Vehicles': 'Number of Vehicles',
#                                          'DayOfWeek': 'Day of Week'})
#                 st.plotly_chart(fig)
#
#             st.plotly_chart(fig)
#
#     # Allow users to update traffic info
#     st.sidebar.title("Update Traffic Information")
#     new_traffic_info = st.sidebar.number_input("Enter new traffic count", min_value=0, value=0)
#     if st.sidebar.button("Update Traffic Count"):
#         # Update logic here
#         if not show_date_checkbox:
#             st.write(
#                 f"New traffic count of {new_traffic_info} at Junctions {junctions} on {year}-{month}-{day} at {hour}:00 updated successfully.")
#         else:
#             st.write(
#                 f"New traffic count of {new_traffic_info} on {year}-{month}-{day} updated successfully.")
#
#     # Allow users to upload photos
#     st.sidebar.title("Upload Photos")
#     uploaded_files = st.sidebar.file_uploader("Choose a photo", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
#     for uploaded_file in uploaded_files:
#         st.sidebar.image(uploaded_file, caption=uploaded_file.name)
# import base64
#
# import streamlit as st
# import pandas as pd
# import numpy as np
# from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
# from sklearn.preprocessing import StandardScaler
# import plotly.express as px
# from PIL import Image
#
# # Load and preprocess data
# @st.cache(allow_output_mutation=True)
# def load_data():
#     # Load the data (replace with your data loading logic)
#     df = pd.read_csv("uncleaned_traffic.csv")
#
#     # Your data preprocessing steps go here
#     # Data cleaning
#     df = df.dropna(subset=['DateTime', 'Junction', 'Vehicles', 'ID'])
#     df = df.drop_duplicates()
#     df['DateTime'] = pd.to_datetime(df['DateTime'])
#     df['Vehicles'] = pd.to_numeric(df['Vehicles'], errors='coerce')  # Ensure 'Vehicles' column is numeric
#     df = df.dropna(subset=['Vehicles'])
#     df['Junction'] = df['Junction'].astype(int)
#     df['ID'] = df['ID'].apply(lambda x: '{:.0f}'.format(x))
#     df = df.drop_duplicates()
#     df = df.sort_values(by=['Junction', 'DateTime'])
#     df.reset_index(drop=True, inplace=True)
#     df.set_index('DateTime', inplace=True)
#
#     # Handling missing values
#     df.fillna(method='ffill', inplace=True)
#
#     # Extracting date-time features
#     df['Year'] = df.index.year
#     df['Month'] = df.index.month
#     df['Day'] = df.index.day
#     df['Hour'] = df.index.hour
#     df['DayOfWeek'] = df.index.dayofweek
#
#     # Adding cyclical features
#     df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
#     df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
#     df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
#     df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
#
#     # Adding holiday feature
#     cal = calendar()
#     holidays = cal.holidays(start=df.index.min(), end=df.index.max())
#     df['Holiday'] = df.index.isin(holidays).astype(int)
#
#     # Feature scaling
#     scaler = StandardScaler()
#     df['Vehicles_scaled'] = scaler.fit_transform(df[['Vehicles']])
#
#     return df
#
# # Load data
# df = load_data()
#
# # Check if 'Year' column exists in the DataFrame
# if 'Year' not in df.columns:
#     st.error("Year column not found in the DataFrame!")
# else:
#     # Initialize junctions variable
#     junctions = []
#
#     # Sidebar filters
#     st.sidebar.title("Traffic Data Filters")
#
#     # Display images of traffic police and department logo
#     st.sidebar.title("Traffic Police")
#     traffic_police_image = Image.open("traffic-police.jpg")
#     st.sidebar.image(traffic_police_image, caption='Traffic Police')
#
#     department_logo_image = Image.open("logo.jpg")
#     st.sidebar.image(department_logo_image, caption='Department Logo')
#
#     # Dropdown select boxes for filtering
#     st.sidebar.title("Date and Time Filters")
#     show_overall_data = st.sidebar.checkbox("Show Overall Data")
#
#     if not show_overall_data:
#         year = st.sidebar.selectbox("Select Year", df['Year'].unique())
#         month = st.sidebar.selectbox("Select Month", df['Month'].unique())
#         day = st.sidebar.selectbox("Select Day", df['Day'].unique())
#         hour = st.sidebar.selectbox("Select Hour", df['Hour'].unique())
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}, Hour: {hour}")
#
#         # Filter data based on user selection
#         filtered_df = df[(df['Year'] == year) & (df['Month'] == month) & (df['Day'] == day) & (df['Hour'] == hour)]
#     else:
#         st.write("Showing overall data")
#
#         # Compute overall data
#         overall_df = df.groupby(['Year', 'Month', 'Day', 'Hour', 'Junction'], as_index=False)['Vehicles'].mean()
#         filtered_df = overall_df
#
#     # Display filtered data
#     st.write(filtered_df)
#
#     # Visualization of filtered data
#     st.title("Historical Traffic Data - Line Chart")
#
#     if not filtered_df.empty:
#         # Group by Hour and Junction to get the average number of vehicles
#         historical_df = filtered_df.groupby(['Hour', 'Junction'], as_index=False)['Vehicles'].mean()
#
#         # Create line chart using Plotly Express
#         fig = px.line(historical_df, x='Hour', y='Vehicles', color='Junction',
#                       title=f'Average Hourly Traffic Data',
#                       labels={'Hour': 'Hour of Day', 'Vehicles': 'Average Number of Vehicles'})
#         st.plotly_chart(fig)
#
#     # Additional data display (if needed)
#     # Example: Displaying a summary or statistics
#     st.title("Additional Data Display")
#     if not filtered_df.empty:
#         st.write(f"Total number of records: {len(filtered_df)}")
#
#     # Allow users to update traffic info
#     st.sidebar.title("Update Traffic Information")
#     new_traffic_info = st.sidebar.number_input("Enter new traffic count", min_value=0, value=0)
#     if st.sidebar.button("Update Traffic Count"):
#         # Update logic here
#         if not show_overall_data:
#             st.write(
#                 f"New traffic count of {new_traffic_info} at {hour}:00 on {month}/{day}/{year} updated successfully.")
#         else:
#             st.write(
#                 f"New traffic count of {new_traffic_info} in overall data updated successfully.")
#
#  # Allow users to upload, update, and delete photos
# st.sidebar.title("Manage Photos")
#
# if st.sidebar.button("Upload Photo"):
#     uploaded_files = st.sidebar.file_uploader("Choose a photo", type=["jpg", "jpeg", "png"])
#     if uploaded_files is not None:
#         uploaded_files.append(uploaded_files)
#
# # Display submitted photos
# if uploaded_files:
#     st.sidebar.title("Submitted Photos")
#     for uploaded_file in uploaded_files:
#         st.sidebar.image(uploaded_file, caption=uploaded_file.name)
#
# # Show only one image
# if uploaded_file:
#     st.sidebar.image(uploaded_file[0], caption=uploaded_file[0].name)
#
# # Option to delete photo
# photo_to_delete = st.sidebar.selectbox("Select photo to delete",
#                                        [uploaded_file.name for uploaded_file in uploaded_files])
# if st.sidebar.button("Delete Photo"):
#     uploaded_files = [file for file in uploaded_file if file.name != photo_to_delete]
#
# # Submit button to process uploaded photos and generate downloadable file
# if st.sidebar.button("Submit Photos"):
#     # Example: Save uploaded files to a CSV with links
#     photos_data = [{'Filename': file.name, 'Link': file.name} for file in uploaded_files]
#
#     # Create a DataFrame from photos_data
#     photos_df = pd.DataFrame(photos_data)
#
#     # Save photos_df to CSV
#     photos_csv = photos_df.to_csv(index=False)
#
#     # Download link for CSV file
#     csv_download_link = f'<a href="data:file/csv;base64,{base64.b64encode(photos_csv.encode()).decode()}" download="uploaded_photos.csv">Download photos data as CSV file</a>'
#     st.sidebar.markdown(csv_download_link, unsafe_allow_html=True)


# import streamlit as st
# import pandas as pd
# import numpy as np
# from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
# from sklearn.preprocessing import StandardScaler
# import plotly.express as px
# from PIL import Image
# import base64
# import io
# import os
#
# # Load and preprocess data
# @st.cache_data
# def load_data():
#     # Load the data (replace with your data loading logic)
#     df = pd.read_csv("uncleaned_traffic.csv")
#
#     # Your data preprocessing steps go here
#     # Data cleaning
#     df = df.dropna(subset=['DateTime', 'Junction', 'Vehicles', 'ID'])
#     df = df.drop_duplicates()
#     df['DateTime'] = pd.to_datetime(df['DateTime'])
#     df['Vehicles'] = pd.to_numeric(df['Vehicles'], errors='coerce')  # Ensure 'Vehicles' column is numeric
#     df = df.dropna(subset=['Vehicles'])
#     df['Junction'] = df['Junction'].astype(int)
#     df['ID'] = df['ID'].apply(lambda x: '{:.0f}'.format(x))
#     df = df.drop_duplicates()
#     df = df.sort_values(by=['Junction', 'DateTime'])
#     df.reset_index(drop=True, inplace=True)
#     df.set_index('DateTime', inplace=True)
#
#     # Handling missing values
#     df.fillna(method='ffill', inplace=True)
#
#     # Extracting date-time features
#     df['Year'] = df.index.year
#     df['Month'] = df.index.month
#     df['Day'] = df.index.day
#     df['Hour'] = df.index.hour
#     df['DayOfWeek'] = df.index.dayofweek
#
#     # Adding cyclical features
#     df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
#     df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
#     df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
#     df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
#
#     # Adding holiday feature
#     cal = calendar()
#     holidays = cal.holidays(start=df.index.min(), end=df.index.max())
#     df['Holiday'] = df.index.isin(holidays).astype(int)
#
#     # Feature scaling
#     scaler = StandardScaler()
#     df['Vehicles_scaled'] = scaler.fit_transform(df[['Vehicles']])
#
#     return df
#
# # Load data
# df = load_data()
#
#
# # Check if 'Year' column exists in the DataFrame
# if 'Year' not in df.columns:
#     st.error("Year column not found in the DataFrame!")
# else:
#     # Initialize junctions variable
#     junctions = []
#
#     # Sidebar filters
#     st.sidebar.title("Traffic Data Filters")
#
#     # Display images of traffic police and department logo
#     st.sidebar.title("Traffic Police")
#     traffic_police_image = Image.open("traffic-police.jpg")
#     st.sidebar.image(traffic_police_image, caption='Traffic Police')
#
#     department_logo_image = Image.open("logo.jpg")
#     st.sidebar.image(department_logo_image, caption='Department Logo')
#
#     # Checkbox for selecting specific date, year, and hour
#     show_date_checkbox = st.sidebar.checkbox("Filter by Date")
#
#     if show_date_checkbox:
#         st.sidebar.title("Date Filters")
#         year = st.sidebar.selectbox("Select Year", df['Year'].unique())
#         month = st.sidebar.selectbox("Select Month", df['Month'].unique())
#         day = st.sidebar.selectbox("Select Day", df['Day'].unique())
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}")
#
#         filtered_df = df[(df['Year'] == year) & (df['Month'] == month) & (df['Day'] == day)]
#     else:
#         st.sidebar.title("Date and Time Filters")
#         year = st.sidebar.selectbox("Select Year", df['Year'].unique())
#         month = st.sidebar.selectbox("Select Month", df['Month'].unique())
#         day = st.sidebar.selectbox("Select Day", df['Day'].unique())
#         hour = st.sidebar.selectbox("Select Hour", df['Hour'].unique())
#         junctions = st.sidebar.multiselect("Select Junction", df['Junction'].unique())
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}, Hour: {hour}, Junctions: {junctions}")
#
#         filtered_df = df[(df['Year'] == year) & (df['Month'] == month) & (df['Day'] == day) & (df['Hour'] == hour) & (
#                     df['Junction'].isin(junctions))]
#
#     # Display filtered data
#     st.write(filtered_df)
#
#     # Visualization of filtered data
#     st.title("Historical Traffic Data - Line Chart")
#
#     if not filtered_df.empty:
#         # Group by Hour and Junction to get the average number of vehicles
#         historical_df = filtered_df.groupby(['Hour', 'Junction'], as_index=False)['Vehicles'].mean()
#
#         # Create line chart using Plotly Express
#         fig = px.line(historical_df, x='Hour', y='Vehicles', color='Junction',
#                       title='Average Hourly Traffic Data',
#                       labels={'Hour': 'Hour of Day', 'Vehicles': 'Average Number of Vehicles'})
#         st.plotly_chart(fig)
#
#     # Additional data display (if needed)
#     # Example: Displaying a summary or statistics
#     st.title("Additional Data Display")
#     if not filtered_df.empty:
#         st.write(f"Total number of records: {len(filtered_df)}")
#
#     # Allow users to update traffic info
#     st.sidebar.title("Update Traffic Information")
#     new_traffic_info = st.sidebar.number_input("Enter new traffic count", min_value=0, value=0)
#     if st.sidebar.button("Update Traffic Count"):
#         # Update logic here
#         if show_date_checkbox:
#             st.write(
#                 f"New traffic count of {new_traffic_info} on {year}-{month}-{day} updated successfully.")
#         else:
#             st.write(
#                 f"New traffic count of {new_traffic_info} at {hour}:00 on {month}/{day}/{year} at Junctions {junctions} updated successfully.")
#
#     # Ensure session state for uploaded files
#     if 'uploaded_files' not in st.session_state:
#         st.session_state.uploaded_files = []
#
#     # Allow users to upload, update, and delete photos
#     st.sidebar.title("Manage Photos")
#
#     uploaded_file = st.sidebar.file_uploader("Choose a photo", type=["jpg", "jpeg", "png"])
#     if uploaded_file is not None:
#         st.session_state.uploaded_files.append(uploaded_file)
#
#     # Display submitted photos
#     if st.session_state.uploaded_files:
#         st.sidebar.title("Submitted Photos")
#         for uploaded_file in st.session_state.uploaded_files:
#             st.sidebar.image(uploaded_file, caption=uploaded_file.name)
#
#     # Show only one image
#     if st.session_state.uploaded_files:
#         st.sidebar.image(st.session_state.uploaded_files[0], caption=st.session_state.uploaded_files[0].name)
#
#     # Option to delete photo
#     photo_to_delete = st.sidebar.selectbox("Select photo to delete", [uploaded_file.name for uploaded_file in st.session_state.uploaded_files])
#     if st.sidebar.button("Delete Photo"):
#         st.session_state.uploaded_files = [file for file in st.session_state.uploaded_files if file.name != photo_to_delete]
#
#     # Submit button to process uploaded photos and generate downloadable file
#     if st.sidebar.button("Submit Photos"):
#         # Example: Save uploaded files to a CSV with links
#         photos_data = [{'Filename': file.name, 'Link': file.name} for file in st.session_state.uploaded_files]
#
#         # Create a DataFrame from photos_data
#         photos_df = pd.DataFrame(photos_data)
#
#         # Save photos_df to CSV
#         photos_csv = photos_df.to_csv(index=False)
#
#         # Download link for CSV file
#         b64 = base64.b64encode(photos_csv.encode()).decode()
#         csv_download_link = f'<a href="data:file/csv;base64,{b64}" download="uploaded_photos.csv">Download photos data as CSV file</a>'
#         st.sidebar.markdown(csv_download_link, unsafe_allow_html=True)


# import streamlit as st
# import pandas as pd
# import numpy as np
# from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
# from sklearn.preprocessing import StandardScaler
# import plotly.express as px
# from PIL import Image
# import base64
# import io
# import os
#
# # Load and preprocess data
# @st.cache_data
# def load_data():
#     # Load the data (replace with your data loading logic)
#     df = pd.read_csv("uncleaned_traffic.csv")
#
#     # Drop rows with missing essential columns
#     df = df.dropna(subset=['DateTime', 'Junction', 'Vehicles', 'ID'])
#
#     # Convert 'DateTime' column to datetime format and set as index
#     df['DateTime'] = pd.to_datetime(df['DateTime'])
#     df.set_index('DateTime', inplace=True)
#
#     # Ensure 'Vehicles' column is numeric
#     df['Vehicles'] = pd.to_numeric(df['Vehicles'], errors='coerce')
#
#     # Drop duplicates
#     df = df.drop_duplicates()
#
#     # Extract features from DateTime
#     df['Year'] = df.index.year
#     df['Month'] = df.index.month
#     df['Day'] = df.index.day
#     df['Hour'] = df.index.hour
#     df['DayOfWeek'] = df.index.dayofweek
#
#     # Handle missing values by forward filling
#     df.fillna(method='ffill', inplace=True)
#
#     # Ensure 'Junction' column is integer
#     df['Junction'] = df['Junction'].astype(int)
#
#     # Ensure 'ID' column is formatted as string without decimal points
#     df['ID'] = df['ID'].apply(lambda x: '{:.0f}'.format(x))
#
#     # Sort by Junction and DateTime
#     df = df.sort_values(by=['Junction', 'DateTime'])
#
#     # Reset index
#     df.reset_index(drop=True, inplace=True)
#
#     # Adding cyclical features for Hour and DayOfWeek
#     df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
#     df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
#     df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
#     df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
#
#     # Adding holiday feature
#     cal = calendar()
#     holidays = cal.holidays(start=df.index.min(), end=df.index.max())
#     df['Holiday'] = df.index.isin(holidays).astype(int)
#
#     # Feature scaling for 'Vehicles' using StandardScaler
#     scaler = StandardScaler()
#     df['Vehicles_scaled'] = scaler.fit_transform(df[['Vehicles']])
#
#     return df
#
# # Load data
# df = load_data()
#
# # Check if 'Year' column exists in the DataFrame
# if 'Year' not in df.columns:
#     st.error("Year column not found in the DataFrame!")
# else:
#     # Initialize junctions variable
#     junctions = []
#
#     # Sidebar filters
#     st.sidebar.title("Traffic Data Filters")
#
#     # Display images of traffic police and department logo
#     st.sidebar.title("Traffic Police")
#     traffic_police_image = Image.open("traffic-police.jpg")
#     st.sidebar.image(traffic_police_image, caption='Traffic Police')
#
#     department_logo_image = Image.open("logo.jpg")
#     st.sidebar.image(department_logo_image, caption='Department Logo')
#
#     # Checkbox for selecting specific date, year, and hour
#     show_date_checkbox = st.sidebar.checkbox("Filter by Date")
#
#     if show_date_checkbox:
#         st.sidebar.title("Date Filters")
#         year = st.sidebar.selectbox("Select Year", df['Year'].unique())
#         month = st.sidebar.selectbox("Select Month", df['Month'].unique())
#         day = st.sidebar.selectbox("Select Day", df['Day'].unique())
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}")
#
#         filtered_df = df[(df['Year'] == year) & (df['Month'] == month) & (df['Day'] == day)]
#     else:
#         st.sidebar.title("Date and Time Filters")
#         year = st.sidebar.selectbox("Select Year", df['Year'].unique())
#         month = st.sidebar.selectbox("Select Month", df['Month'].unique())
#         day = st.sidebar.selectbox("Select Day", df['Day'].unique())
#         hour = st.sidebar.selectbox("Select Hour", df['Hour'].unique())
#         junctions = st.sidebar.multiselect("Select Junction", df['Junction'].unique())
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}, Hour: {hour}, Junctions: {junctions}")
#
#         filtered_df = df[(df['Year'] == year) & (df['Month'] == month) & (df['Day'] == day) & (df['Hour'] == hour) & (
#                     df['Junction'].isin(junctions))]
#
#     # Display filtered data
#     st.write(filtered_df)
#
#     # Visualization of filtered data
#     st.title("Historical Traffic Data - Line Chart")
#
#     if not filtered_df.empty:
#         # Group by Hour and Junction to get the average number of vehicles
#         historical_df = filtered_df.groupby(['Hour', 'Junction'], as_index=False)['Vehicles'].mean()
#
#         # Create line chart using Plotly Express
#         fig = px.line(historical_df, x='Hour', y='Vehicles', color='Junction',
#                       title='Average Hourly Traffic Data',
#                       labels={'Hour': 'Hour of Day', 'Vehicles': 'Average Number of Vehicles'})
#         st.plotly_chart(fig)
#
#     # Additional data display (if needed)
#     # Example: Displaying a summary or statistics
#     st.title("Additional Data Display")
#     if not filtered_df.empty:
#         st.write(f"Total number of records: {len(filtered_df)}")
#
#     # Allow users to update traffic info
#     st.sidebar.title("Update Traffic Information")
#     new_traffic_info = st.sidebar.number_input("Enter new traffic count", min_value=0, value=0)
#     if st.sidebar.button("Update Traffic Count"):
#         # Update logic here
#         if show_date_checkbox:
#             st.write(
#                 f"New traffic count of {new_traffic_info} on {year}-{month}-{day} updated successfully.")
#         else:
#             st.write(
#                 f"New traffic count of {new_traffic_info} at {hour}:00 on {month}/{day}/{year} at Junctions {junctions} updated successfully.")
#
#     # Ensure session state for uploaded files
#     if 'uploaded_files' not in st.session_state:
#         st.session_state.uploaded_files = []
#
#     # Allow users to upload, update, and delete photos
#     st.sidebar.title("Manage Photos")
#
#     uploaded_file = st.sidebar.file_uploader("Choose a photo", type=["jpg", "jpeg", "png"])
#     if uploaded_file is not None:
#         st.session_state.uploaded_files.append(uploaded_file)
#
#     # Display submitted photos
#     if st.session_state.uploaded_files:
#         st.sidebar.title("Submitted Photos")
#         for uploaded_file in st.session_state.uploaded_files:
#             st.sidebar.image(uploaded_file, caption=uploaded_file.name)
#
#     # Show only one image
#     if st.session_state.uploaded_files:
#         st.sidebar.image(st.session_state.uploaded_files[0], caption=st.session_state.uploaded_files[0].name)
#
#     # Option to delete photo
#     photo_to_delete = st.sidebar.selectbox("Select photo to delete", [uploaded_file.name for uploaded_file in st.session_state.uploaded_files])
#     if st.sidebar.button("Delete Photo"):
#         st.session_state.uploaded_files = [file for file in st.session_state.uploaded_files if file.name != photo_to_delete]
#
#     # Submit button to process uploaded photos and generate downloadable file
#     if st.sidebar.button("Submit Photos"):
#         # Example: Save uploaded files to a CSV with links
#         photos_data = [{'Filename': file.name, 'Link': file.name} for file in st.session_state.uploaded_files]
#
#         # Create a DataFrame from photos_data
#         photos_df = pd.DataFrame(photos_data)
#
#         # Save photos_df to CSV
#         photos_csv = photos_df.to_csv(index=False)
#
#         # Download link for CSV file
#         b64 = base64.b64encode(photos_csv.encode()).decode()
#         csv_download_link = f'<a href="data:file/csv;base64,{b64}" download="uploaded_photos.csv">Download photos data as CSV file</a>'
#         st.sidebar.markdown(csv_download_link, unsafe_allow_html=True)
#
#     # Overall Traffic Data Overview
#     st.title("Overall Traffic Data Overview")
#
#     # Calculate total number of vehicles per month
#     overall_data = df.groupby('Month')['Vehicles'].sum().reset_index()
#
#     # Plot bar chart using Plotly Express
#     fig_overall = px.bar(overall_data, x='Month', y='Vehicles',
#                          title='Total Number of Vehicles per Month',
#                          labels={'Month': 'Month', 'Vehicles': 'Total Vehicles'})
#     st.plotly_chart(fig_overall)
#
#
# import streamlit as st
# import pandas as pd
# import numpy as np
# from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
# from sklearn.preprocessing import StandardScaler
# import plotly.express as px
# from PIL import Image
# import base64
#
#
# # Load and preprocess data
# @st.cache_data
# def load_data():
#     # Load the data (replace with your data loading logic)
#     df = pd.read_csv("uncleaned_traffic.csv")
#
#     # Your data preprocessing steps go here
#     # Data cleaning
#     df = df.dropna(subset=['DateTime', 'Junction', 'Vehicles', 'ID'])
#     df = df.drop_duplicates()
#     df['DateTime'] = pd.to_datetime(df['DateTime'])
#     df['Vehicles'] = pd.to_numeric(df['Vehicles'], errors='coerce')  # Ensure 'Vehicles' column is numeric
#     df = df.dropna(subset=['Vehicles'])
#     df['Junction'] = df['Junction'].astype(int)
#     df['ID'] = df['ID'].apply(lambda x: '{:.0f}'.format(x))
#     df = df.drop_duplicates()
#     df = df.sort_values(by=['Junction', 'DateTime'])
#     df.reset_index(drop=True, inplace=True)
#     df.set_index('DateTime', inplace=True)
#
#     # Handling missing values
#     df.fillna(method='ffill', inplace=True)
#
#     # Extracting date-time features
#     df['Year'] = df.index.year
#     df['Month'] = df.index.month
#     df['Day'] = df.index.day
#     df['Hour'] = df.index.hour
#     df['DayOfWeek'] = df.index.dayofweek
#
#     # Adding cyclical features
#     df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
#     df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
#     df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
#     df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
#
#     # Adding holiday feature
#     cal = calendar()
#     holidays = cal.holidays(start=df.index.min(), end=df.index.max())
#     df['Holiday'] = df.index.isin(holidays).astype(int)
#
#     # Feature scaling
#     scaler = StandardScaler()
#     df['Vehicles_scaled'] = scaler.fit_transform(df[['Vehicles']])
#
#     return df
#
#
# # Function to display Traffic Data tab
# def display_traffic_data(df):
#     st.title("Traffic Data")
#
#     # Check if 'Year' column exists in the DataFrame
#     if 'Year' not in df.columns:
#         st.error("Year column not found in the DataFrame!")
#         return
#
#     # Initialize junctions variable
#     junctions = []
#
#     # Sidebar filters
#     st.sidebar.title("Traffic Data Filters")
#
#     # Checkbox for selecting specific date, year, and hour
#     show_date_checkbox = st.sidebar.checkbox("Filter by Date")
#
#     if show_date_checkbox:
#         st.sidebar.title("Date Filters")
#         year = st.sidebar.selectbox("Select Year", np.append(["Overall"], df['Year'].unique()))
#         month = st.sidebar.selectbox("Select Month", np.append(["Overall"], df['Month'].unique()))
#         day = st.sidebar.selectbox("Select Day", np.append(["Overall"], df['Day'].unique()))
#
#         if year == "Overall":
#             filtered_df = df
#         else:
#             filtered_df = df[df['Year'] == int(year)]
#
#         if month != "Overall":
#             filtered_df = filtered_df[filtered_df['Month'] == int(month)]
#
#         if day != "Overall":
#             filtered_df = filtered_df[filtered_df['Day'] == int(day)]
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}")
#
#     else:
#         st.sidebar.title("Date and Time Filters")
#         year = st.sidebar.selectbox("Select Year", np.append(["Overall"], df['Year'].unique()))
#         month = st.sidebar.selectbox("Select Month", np.append(["Overall"], df['Month'].unique()))
#         day = st.sidebar.selectbox("Select Day", np.append(["Overall"], df['Day'].unique()))
#         hour = st.sidebar.selectbox("Select Hour", np.append(["Overall"], df['Hour'].unique()))
#         junctions = st.sidebar.multiselect("Select Junction", df['Junction'].unique(), default=df['Junction'].unique())
#
#         filtered_df = df
#
#         if year != "Overall":
#             filtered_df = filtered_df[filtered_df['Year'] == int(year)]
#
#         if month != "Overall":
#             filtered_df = filtered_df[filtered_df['Month'] == int(month)]
#
#         if day != "Overall":
#             filtered_df = filtered_df[filtered_df['Day'] == int(day)]
#
#         if hour != "Overall":
#             filtered_df = filtered_df[filtered_df['Hour'] == int(hour)]
#
#         if junctions:
#             filtered_df = filtered_df[filtered_df['Junction'].isin(junctions)]
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}, Hour: {hour}, Junctions: {junctions}")
#
#     # Display filtered data
#     st.write(filtered_df)
#
#     # Visualization of filtered data
#     st.title("Historical Traffic Data - Line Chart")
#
#     if not filtered_df.empty:
#         # Group by Hour and Junction to get the average number of vehicles
#         historical_df = filtered_df.groupby(['Hour', 'Junction'], as_index=False)['Vehicles'].mean()
#
#         # Create line chart using Plotly Express
#         fig = px.line(historical_df, x='Hour', y='Vehicles', color='Junction',
#                       title='Average Hourly Traffic Data',
#                       labels={'Hour': 'Hour of Day', 'Vehicles': 'Average Number of Vehicles'})
#         st.plotly_chart(fig)
#
#     # Additional data display (if needed)
#     # Example: Displaying a summary or statistics
#     st.title("Additional Data Display")
#     if not filtered_df.empty:
#         st.write(f"Total number of records: {len(filtered_df)}")
#
#
# # Function to display Upload Photos tab
# def display_upload_photos():
#     st.title("Upload Photos")
#
#     # Allow users to upload photos
#     uploaded_files = st.file_uploader("Choose a photo", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
#
#     # Display uploaded photos
#     if uploaded_files:
#         for uploaded_file in uploaded_files:
#             st.image(uploaded_file, caption=uploaded_file.name)
#
#         # Save uploaded files to session state
#         st.session_state.uploaded_photos.extend(uploaded_files)
#
#         # Display success message
#         st.success("Photos uploaded successfully!")
#
#
# # Function to display Settings tab
# def display_settings():
#     st.title("Settings")
#
#     # Placeholder for settings content
#     st.write("Settings page content goes here.")
#
#
# # Main function to run the Streamlit app
# def main():
#     # Set page title and icon
#     st.set_page_config(page_title="Traffic Data Analysis", page_icon=":car:")
#
#     # Load data
#     df = load_data()
#
#     # Create sidebar navigation
#     st.sidebar.title("Navigation")
#     selected_tab = st.sidebar.radio("Go to", ["Traffic Data", "Upload Photos", "Settings"])
#
#     # Display content based on selected tab
#     if selected_tab == "Traffic Data":
#         display_traffic_data(df)
#     elif selected_tab == "Upload Photos":
#         display_upload_photos()
#     elif selected_tab == "Settings":
#         display_settings()
#
# #
# # Run the app
# if __name__ == "__main__":
#     main()
# import streamlit as st
# import pandas as pd
# import numpy as np
# from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
# from sklearn.preprocessing import StandardScaler
# import plotly.express as px
# from PIL import Image
# import base64
#
#
# # Load and preprocess data
# @st.cache_data
# def load_data():
#     # Load the data (replace with your data loading logic)
#     df = pd.read_csv("uncleaned_traffic.csv")
#
#     # Your data preprocessing steps go here
#     # Data cleaning
#     df = df.dropna(subset=['DateTime', 'Junction', 'Vehicles', 'ID'])
#     df = df.drop_duplicates()
#     df['DateTime'] = pd.to_datetime(df['DateTime'])
#     df['Vehicles'] = pd.to_numeric(df['Vehicles'], errors='coerce')  # Ensure 'Vehicles' column is numeric
#     df = df.dropna(subset=['Vehicles'])
#     df['Junction'] = df['Junction'].astype(int)
#     df['ID'] = df['ID'].apply(lambda x: '{:.0f}'.format(x))
#     df = df.drop_duplicates()
#     df = df.sort_values(by=['Junction', 'DateTime'])
#     df.reset_index(drop=True, inplace=True)
#     df.set_index('DateTime', inplace=True)
#
#     # Handling missing values
#     df.fillna(method='ffill', inplace=True)
#
#     # Extracting date-time features
#     df['Year'] = df.index.year
#     df['Month'] = df.index.month
#     df['Day'] = df.index.day
#     df['Hour'] = df.index.hour
#     df['DayOfWeek'] = df.index.dayofweek
#
#     # Adding cyclical features
#     df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
#     df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
#     df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
#     df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
#
#     # Adding holiday feature
#     cal = calendar()
#     holidays = cal.holidays(start=df.index.min(), end=df.index.max())
#     df['Holiday'] = df.index.isin(holidays).astype(int)
#
#     # Feature scaling
#     scaler = StandardScaler()
#     df['Vehicles_scaled'] = scaler.fit_transform(df[['Vehicles']])
#
#     return df
#
#
# # Function to display Traffic Data tab
# def display_traffic_data(df):
#     st.title("Traffic Data Analysis")
#
#     # Check if 'Year' column exists in the DataFrame
#     if 'Year' not in df.columns:
#         st.error("Year column not found in the DataFrame!")
#         return
#
#     # Initialize junctions variable
#     junctions = []
#
#     # Sidebar filters
#     st.sidebar.title("Traffic Data Filters")
#
#     # Checkbox for selecting specific date, year, and hour
#     show_date_checkbox = st.sidebar.checkbox("Filter by Date")
#
#     if show_date_checkbox:
#         st.sidebar.title("Date Filters")
#         year = st.sidebar.selectbox("Select Year", np.append(["Overall"], df['Year'].unique()))
#         month = st.sidebar.selectbox("Select Month", np.append(["Overall"], df['Month'].unique()))
#         day = st.sidebar.selectbox("Select Day", np.append(["Overall"], df['Day'].unique()))
#
#         if year == "Overall":
#             filtered_df = df
#         else:
#             filtered_df = df[df['Year'] == int(year)]
#
#         if month != "Overall":
#             filtered_df = filtered_df[filtered_df['Month'] == int(month)]
#
#         if day != "Overall":
#             filtered_df = filtered_df[filtered_df['Day'] == int(day)]
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}")
#
#     else:
#         st.sidebar.title("Date and Time Filters")
#         year = st.sidebar.selectbox("Select Year", np.append(["Overall"], df['Year'].unique()))
#         month = st.sidebar.selectbox("Select Month", np.append(["Overall"], df['Month'].unique()))
#         day = st.sidebar.selectbox("Select Day", np.append(["Overall"], df['Day'].unique()))
#         hour = st.sidebar.selectbox("Select Hour", np.append(["Overall"], df['Hour'].unique()))
#         junctions = st.sidebar.multiselect("Select Junction", df['Junction'].unique(), default=df['Junction'].unique())
#
#         filtered_df = df
#
#         if year != "Overall":
#             filtered_df = filtered_df[filtered_df['Year'] == int(year)]
#
#         if month != "Overall":
#             filtered_df = filtered_df[filtered_df['Month'] == int(month)]
#
#         if day != "Overall":
#             filtered_df = filtered_df[filtered_df['Day'] == int(day)]
#
#         if hour != "Overall":
#             filtered_df = filtered_df[filtered_df['Hour'] == int(hour)]
#
#         if junctions:
#             filtered_df = filtered_df[filtered_df['Junction'].isin(junctions)]
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}, Hour: {hour}, Junctions: {junctions}")
#
#     # Display filtered data
#     st.write(filtered_df)
#
#     # Visualization of filtered data
#     st.title("Historical Traffic Data - Line Chart")
#
#     if not filtered_df.empty:
#         # Group by Hour and Junction to get the average number of vehicles
#         historical_df = filtered_df.groupby(['Hour', 'Junction'], as_index=False)['Vehicles'].mean()
#
#         # Create line chart using Plotly Express
#         fig = px.line(historical_df, x='Hour', y='Vehicles', color='Junction',
#                       title='Average Hourly Traffic Data',
#                       labels={'Hour': 'Hour of Day', 'Vehicles': 'Average Number of Vehicles'})
#         st.plotly_chart(fig)
#
#
# # Function to display Upload Photos tab
# def display_upload_photos():
#     st.title("Upload Photos")
#
#     # Allow users to upload photos
#     uploaded_files = st.file_uploader("Choose a photo", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
#
#     # Display uploaded photos
#     if uploaded_files:
#         for uploaded_file in uploaded_files:
#             st.image(uploaded_file, caption=uploaded_file.name)
#
#         # Save uploaded files to session state
#         st.session_state.uploaded_photos.extend(uploaded_files)
#
#         # Display success message
#         st.success("Photos uploaded successfully!")
#
#
# # Function to display Settings tab
# def display_settings():
#     st.title("Settings")
#
#     # Placeholder for settings content
#     st.write("Settings page content goes here.")
#
#
# # Main function to run the Streamlit app
# def main():
#     # Set page title and icon
#     st.set_page_config(page_title="Traffic Data Analysis", page_icon=":car:")
#
#     # Load data
#     df = load_data()
#
#     # Create sidebar navigation
#     st.sidebar.title("Navigation")
#     selected_tab = st.sidebar.radio("Go to", ["Traffic Data", "Upload Photos", "Settings"])
#
#     # Display content based on selected tab
#     if selected_tab == "Traffic Data":
#         display_traffic_data(df)
#     elif selected_tab == "Upload Photos":
#         display_upload_photos()
#     elif selected_tab == "Settings":
#         display_settings()
#
#
# # Run the app
# if __name__ == "__main__":
#     main()

#
# import streamlit as st
# import pandas as pd
# import numpy as np
# from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
# from sklearn.preprocessing import StandardScaler
# import plotly.express as px
# from PIL import Image
# import base64
#
#
# # Load and preprocess data
# @st.cache_data
# def load_data():
#     # Load the data (replace with your data loading logic)
#     df = pd.read_csv("uncleaned_traffic.csv")
#
#     # Your data preprocessing steps go here
#     # Data cleaning
#     df = df.dropna(subset=['DateTime', 'Junction', 'Vehicles', 'ID'])
#     df = df.drop_duplicates()
#     df['DateTime'] = pd.to_datetime(df['DateTime'])
#     df['Vehicles'] = pd.to_numeric(df['Vehicles'], errors='coerce')  # Ensure 'Vehicles' column is numeric
#     df = df.dropna(subset=['Vehicles'])
#     df['Junction'] = df['Junction'].astype(int)
#     df['ID'] = df['ID'].apply(lambda x: '{:.0f}'.format(x))
#     df = df.drop_duplicates()
#     df = df.sort_values(by=['Junction', 'DateTime'])
#     df.reset_index(drop=True, inplace=True)
#     df.set_index('DateTime', inplace=True)
#
#     # Handling missing values
#     df.fillna(method='ffill', inplace=True)
#
#     # Extracting date-time features
#     df['Year'] = df.index.year
#     df['Month'] = df.index.month
#     df['Day'] = df.index.day
#     df['Hour'] = df.index.hour
#     df['DayOfWeek'] = df.index.dayofweek
#
#     # Adding cyclical features
#     df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
#     df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
#     df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
#     df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
#
#     # Adding holiday feature
#     cal = calendar()
#     holidays = cal.holidays(start=df.index.min(), end=df.index.max())
#     df['Holiday'] = df.index.isin(holidays).astype(int)
#
#     # Feature scaling
#     scaler = StandardScaler()
#     df['Vehicles_scaled'] = scaler.fit_transform(df[['Vehicles']])
#
#     return df
#
#
# # Initialize session state
# if 'uploaded_photos' not in st.session_state:
#     st.session_state.uploaded_photos = []
#
#
# # Function to display Traffic Data tab
# def display_traffic_data(df):
#     st.title("Traffic Data Analysis")
#
#     # Check if 'Year' column exists in the DataFrame
#     if 'Year' not in df.columns:
#         st.error("Year column not found in the DataFrame!")
#         return
#
#     # Initialize junctions variable
#     junctions = []
#
#     # Sidebar filters
#     st.sidebar.title("Traffic Data Filters")
#
#     # Checkbox for selecting specific date, year, and hour
#     show_date_checkbox = st.sidebar.checkbox("Filter by Date")
#
#     if show_date_checkbox:
#         st.sidebar.title("Date Filters")
#         year = st.sidebar.selectbox("Select Year", np.append(["Overall"], df['Year'].unique()))
#         month = st.sidebar.selectbox("Select Month", np.append(["Overall"], df['Month'].unique()))
#         day = st.sidebar.selectbox("Select Day", np.append(["Overall"], df['Day'].unique()))
#
#         if year == "Overall":
#             filtered_df = df
#         else:
#             filtered_df = df[df['Year'] == int(year)]
#
#         if month != "Overall":
#             filtered_df = filtered_df[filtered_df['Month'] == int(month)]
#
#         if day != "Overall":
#             filtered_df = filtered_df[filtered_df['Day'] == int(day)]
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}")
#
#     else:
#         st.sidebar.title("Date and Time Filters")
#         year = st.sidebar.selectbox("Select Year", np.append(["Overall"], df['Year'].unique()))
#         month = st.sidebar.selectbox("Select Month", np.append(["Overall"], df['Month'].unique()))
#         day = st.sidebar.selectbox("Select Day", np.append(["Overall"], df['Day'].unique()))
#         hour = st.sidebar.selectbox("Select Hour", np.append(["Overall"], df['Hour'].unique()))
#         junctions = st.sidebar.multiselect("Select Junction", df['Junction'].unique(), default=df['Junction'].unique())
#
#         filtered_df = df
#
#         if year != "Overall":
#             filtered_df = filtered_df[filtered_df['Year'] == int(year)]
#
#         if month != "Overall":
#             filtered_df = filtered_df[filtered_df['Month'] == int(month)]
#
#         if day != "Overall":
#             filtered_df = filtered_df[filtered_df['Day'] == int(day)]
#
#         if hour != "Overall":
#             filtered_df = filtered_df[filtered_df['Hour'] == int(hour)]
#
#         if junctions:
#             filtered_df = filtered_df[filtered_df['Junction'].isin(junctions)]
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}, Hour: {hour}, Junctions: {junctions}")
#
#     # Display filtered data
#     st.write(filtered_df)
#
#     # Visualization of filtered data
#     st.title("Historical Traffic Data - Line Chart")
#
#     if not filtered_df.empty:
#         # Group by Hour and Junction to get the average number of vehicles
#         historical_df = filtered_df.groupby(['Hour', 'Junction'], as_index=False)['Vehicles'].mean()
#
#         # Create line chart using Plotly Express
#         fig = px.line(historical_df, x='Hour', y='Vehicles', color='Junction',
#                       title='Average Hourly Traffic Data',
#                       labels={'Hour': 'Hour of Day', 'Vehicles': 'Average Number of Vehicles'})
#         st.plotly_chart(fig)
#
#
# # Function to display Upload Photos tab
# def display_upload_photos():
#     st.title("Upload Photos")
#
#     # Allow users to upload photos
#     uploaded_files = st.file_uploader("Choose a photo", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
#
#     # Display uploaded photos
#     if uploaded_files:
#         for uploaded_file in uploaded_files:
#             st.image(uploaded_file, caption=uploaded_file.name)
#
#         # Append uploaded files to session state
#         st.session_state.uploaded_photos.extend(uploaded_files)
#
#         # Display success message
#         st.success("Photos uploaded successfully!")
#
#     # Display option to download uploaded photos
#     if st.session_state.uploaded_photos:
#         st.title("Download Uploaded Photos")
#
#         # Display links for downloading uploaded photos
#         for i, photo in enumerate(st.session_state.uploaded_photos, start=1):
#             photo_name = f"Uploaded Photo {i}"
#             st.markdown(f"**{photo_name}**: ")
#             st.markdown(get_image_download_link(photo, photo_name), unsafe_allow_html=True)
#
#
# # Function to get download link for an image
# def get_image_download_link(img, text):
#     buffered = BytesIO()
#     img.save(buffered, format="JPEG")
#     img_str = base64.b64encode(buffered.getvalue()).decode()
#     href = f'<a href="data:file/jpg;base64,{img_str}" download="{text}.jpg">Download {text}</a>'
#     return href
#
#
# # Function to display Settings tab
# def display_settings():
#     st.title("Settings")
#
#     # Placeholder for settings content
#     st.write("Settings page content goes here.")
#
#     st.header("Visualization Settings")
#     # Example: Adjusting line chart granularity
#     line_chart_granularity = st.selectbox("Line Chart Granularity", ["Hour", "Day", "Month"])
#     st.write(f"Selected Line Chart Granularity: {line_chart_granularity}")
#
#     # Example: Adjusting color scheme
#     color_scheme = st.selectbox("Color Scheme", ["Plotly Express", "Streamlit Default", "Custom"])
#     st.write(f"Selected Color Scheme: {color_scheme}")
#
#     # Example: Adjusting plot size
#     plot_size = st.slider("Plot Size", min_value=200, max_value=1000, value=500, step=50)
#     st.write(f"Selected Plot Size: {plot_size}")
#
#     st.header("Data Filtering Defaults")
#     # Example: Default year selection
#     default_year = st.selectbox("Default Year", np.append(["Overall"], df['Year'].unique()))
#     st.write(f"Default Year: {default_year}")
#
#     # Example: Default junction selection
#     default_junction = st.multiselect("Default Junctions", df['Junction'].unique(), default=df['Junction'].unique())
#     st.write(f"Default Junctions: {default_junction}")
#
#
# # Main function to run the Streamlit app
# def main():
#     # Set page title and icon
#     st.set_page_config(page_title="Traffic Data Analysis", page_icon=":car:")
#
#     # Load data
#     df = load_data()
#
#     # Create sidebar navigation
#     st.sidebar.title("Navigation")
#     selected_tab = st.sidebar.radio("Go to", ["Traffic Data", "Upload Photos", "Settings"])
#
#     # Display content based on selected tab
#     if selected_tab == "Traffic Data":
#         display_traffic_data(df)
#     elif selected_tab == "Upload Photos":
#         display_upload_photos()
#     elif selected_tab == "Settings":
#         display_settings()
#
#
# # Run the app
# if __name__ == "__main__":
#     main()
#
# import streamlit as st
# import pandas as pd
# import numpy as np
# from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
# from sklearn.preprocessing import StandardScaler
# import plotly.express as px
# from PIL import Image
# import base64
# from io import BytesIO  # Import BytesIO for handling bytes data
#
#
# # Load and preprocess data
# @st.cache_data
# def load_data():
#     # Load the data (replace with your data loading logic)
#     df = pd.read_csv("uncleaned_traffic.csv")
#
#     # Your data preprocessing steps go here
#     # Data cleaning
#     df = df.dropna(subset=['DateTime', 'Junction', 'Vehicles', 'ID'])
#     df = df.drop_duplicates()
#     df['DateTime'] = pd.to_datetime(df['DateTime'])
#     df['Vehicles'] = pd.to_numeric(df['Vehicles'], errors='coerce')  # Ensure 'Vehicles' column is numeric
#     df = df.dropna(subset=['Vehicles'])
#     df['Junction'] = df['Junction'].astype(int)
#     df['ID'] = df['ID'].apply(lambda x: '{:.0f}'.format(x))
#     df = df.drop_duplicates()
#     df = df.sort_values(by=['Junction', 'DateTime'])
#     df.reset_index(drop=True, inplace=True)
#     df.set_index('DateTime', inplace=True)
#
#     # Handling missing values
#     df.fillna(method='ffill', inplace=True)
#
#     # Extracting date-time features
#     df['Year'] = df.index.year
#     df['Month'] = df.index.month
#     df['Day'] = df.index.day
#     df['Hour'] = df.index.hour
#     df['DayOfWeek'] = df.index.dayofweek
#
#     # Adding cyclical features
#     df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
#     df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
#     df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
#     df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
#
#     # Adding holiday feature
#     cal = calendar()
#     holidays = cal.holidays(start=df.index.min(), end=df.index.max())
#     df['Holiday'] = df.index.isin(holidays).astype(int)
#
#     # Feature scaling
#     scaler = StandardScaler()
#     df['Vehicles_scaled'] = scaler.fit_transform(df[['Vehicles']])
#
#     return df
#
#
# # Initialize session state
# if 'uploaded_photos' not in st.session_state:
#     st.session_state.uploaded_photos = []
#
#
# # Main function to run the Streamlit app
# def main():
#     # Set page title and icon
#     st.set_page_config(page_title="Traffic Data Analysis", page_icon=":car:")
#
#     # Load data
#     df = load_data()
#
#     # Create sidebar navigation
#     st.sidebar.title("Navigation")
#     selected_tab = st.sidebar.radio("Go to", ["Traffic Data", "Upload Photos", "Settings"])
#
#     # Display selected tab content
#     if selected_tab == "Traffic Data":
#         display_traffic_data(df)
#     elif selected_tab == "Upload Photos":
#         display_upload_photos()
#     elif selected_tab == "Settings":
#         display_settings(df)
#
#
# # Function to display Traffic Data tab
# def display_traffic_data(df):
#     st.title("Traffic Data Analysis")
#
#     # Check if 'Year' column exists in the DataFrame
#     if 'Year' not in df.columns:
#         st.error("Year column not found in the DataFrame!")
#         return
#
#     # Initialize junctions variable
#     junctions = []
#
#     # Sidebar filters
#     st.sidebar.title("Traffic Data Filters")
#
#     # Checkbox for selecting specific date, year, and hour
#     show_date_checkbox = st.sidebar.checkbox("Filter by Date")
#
#     if show_date_checkbox:
#         st.sidebar.title("Date Filters")
#         year = st.sidebar.selectbox("Select Year", np.append(["Overall"], df['Year'].unique()))
#         month = st.sidebar.selectbox("Select Month", np.append(["Overall"], df['Month'].unique()))
#         day = st.sidebar.selectbox("Select Day", np.append(["Overall"], df['Day'].unique()))
#
#         if year == "Overall":
#             filtered_df = df
#         else:
#             filtered_df = df[df['Year'] == int(year)]
#
#         if month != "Overall":
#             filtered_df = filtered_df[filtered_df['Month'] == int(month)]
#
#         if day != "Overall":
#             filtered_df = filtered_df[filtered_df['Day'] == int(day)]
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}")
#
#     else:
#         st.sidebar.title("Date and Time Filters")
#         year = st.sidebar.selectbox("Select Year", np.append(["Overall"], df['Year'].unique()))
#         month = st.sidebar.selectbox("Select Month", np.append(["Overall"], df['Month'].unique()))
#         day = st.sidebar.selectbox("Select Day", np.append(["Overall"], df['Day'].unique()))
#         hour = st.sidebar.selectbox("Select Hour", np.append(["Overall"], df['Hour'].unique()))
#         junctions = st.sidebar.multiselect("Select Junction", df['Junction'].unique(), default=df['Junction'].unique())
#
#         filtered_df = df
#
#         if year != "Overall":
#             filtered_df = filtered_df[filtered_df['Year'] == int(year)]
#
#         if month != "Overall":
#             filtered_df = filtered_df[filtered_df['Month'] == int(month)]
#
#         if day != "Overall":
#             filtered_df = filtered_df[filtered_df['Day'] == int(day)]
#
#         if hour != "Overall":
#             filtered_df = filtered_df[filtered_df['Hour'] == int(hour)]
#
#         if junctions:
#             filtered_df = filtered_df[filtered_df['Junction'].isin(junctions)]
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}, Hour: {hour}, Junctions: {junctions}")
#
#     # Display filtered data
#     st.write(filtered_df)
#
#     # Visualization of filtered data
#     st.title("Historical Traffic Data - Line Chart")
#
#     if not filtered_df.empty:
#         # Group by Hour and Junction to get the average number of vehicles
#         historical_df = filtered_df.groupby(['Hour', 'Junction'], as_index=False)['Vehicles'].mean()
#
#         # Create line chart using Plotly Express
#         fig = px.line(historical_df, x='Hour', y='Vehicles', color='Junction',
#                       title='Average Hourly Traffic Data',
#                       labels={'Hour': 'Hour of Day', 'Vehicles': 'Average Number of Vehicles'})
#         st.plotly_chart(fig)
#
#
# # Function to display Upload Photos tab
# def display_upload_photos():
#     st.title("Upload Photos")
#
#     # Allow users to upload photos
#     uploaded_files = st.file_uploader("Choose a photo", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
#
#     # Display uploaded photos
#     if uploaded_files:
#         for uploaded_file in uploaded_files:
#             st.image(uploaded_file, caption=uploaded_file.name)
#
#         # Append uploaded files to session state
#         st.session_state.uploaded_photos.extend(uploaded_files)
#
#         # Display success message
#         st.success("Photos uploaded successfully!")
#
#     # Display option to download uploaded photos
#     if st.session_state.uploaded_photos:
#         st.title("Download Uploaded Photos")
#
#         # Display links for downloading uploaded photos
#         for i, photo in enumerate(st.session_state.uploaded_photos, start=1):
#             photo_name = f"Uploaded Photo {i}"
#             st.markdown(f"**{photo_name}**: ")
#             st.markdown(get_image_download_link(photo, photo_name), unsafe_allow_html=True)
#
#
# # Function to get download link for an image
# def get_image_download_link(img, text):
#     buffered = BytesIO()
#     img.save(buffered, format="JPEG")
#     img_str = base64.b64encode(buffered.getvalue()).decode()
#     href = f'<a href="data:file/jpg;base64,{img_str}" download="{text}.jpg">Download {text}</a>'
#     return href
#
#
# # Function to display Settings tab
# def display_settings(df):
#     st.title("Settings")
#
#     st.header("Visualization Settings")
#     # Example: Adjusting line chart granularity
#     line_chart_granularity = st.selectbox("Line Chart Granularity", ["Hour", "Day", "Month"])
#     st.write(f"Selected Line Chart Granularity: {line_chart_granularity}")
#
#     # Example: Adjusting color scheme
#     color_scheme = st.selectbox("Color Scheme", ["Plotly Express", "Streamlit Default", "Custom"])
#     st.write(f"Selected Color Scheme: {color_scheme}")
#
#     # Example: Adjusting plot size
#     plot_size = st.slider("Plot Size", min_value=200, max_value=1000, value=500, step=50)
#     st.write(f"Selected Plot Size: {plot_size}")
#
#     st.header("Data Filtering Defaults")
#     # Example: Default year selection
#     default_year = st.selectbox("Default Year", np.append(["Overall"], df['Year'].unique()))
#     st.write(f"Default Year: {default_year}")
#
#     # Example: Default junction selection
#     default_junction = st.multiselect("Default Junction", df['Junction'].unique(), default=df['Junction'].unique())
#     st.write(f"Default Junctions: {default_junction}")
#
#
# # Run the app
# if __name__ == "__main__":
#     main()


# import streamlit as st
# import pandas as pd
# import numpy as np
# from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
# from sklearn.preprocessing import StandardScaler
# import plotly.express as px
# from PIL import Image
# import base64
# from io import BytesIO  # Import BytesIO for handling bytes data
#
#
# # Load and preprocess data
# @st.cache
# def load_data():
#     # Load the data (replace with your data loading logic)
#     df = pd.read_csv("uncleaned_traffic.csv")
#
#     # Your data preprocessing steps go here
#     # Data cleaning
#     df = df.dropna(subset=['DateTime', 'Junction', 'Vehicles', 'ID'])
#     df = df.drop_duplicates()
#     df['DateTime'] = pd.to_datetime(df['DateTime'])
#     df['Vehicles'] = pd.to_numeric(df['Vehicles'], errors='coerce')  # Ensure 'Vehicles' column is numeric
#     df = df.dropna(subset=['Vehicles'])
#     df['Junction'] = df['Junction'].astype(int)
#     df['ID'] = df['ID'].apply(lambda x: '{:.0f}'.format(x))
#     df = df.drop_duplicates()
#     df = df.sort_values(by=['Junction', 'DateTime'])
#     df.reset_index(drop=True, inplace=True)
#     df.set_index('DateTime', inplace=True)
#
#     # Handling missing values
#     df.fillna(method='ffill', inplace=True)
#
#     # Extracting date-time features
#     df['Year'] = df.index.year
#     df['Month'] = df.index.month
#     df['Day'] = df.index.day
#     df['Hour'] = df.index.hour
#     df['DayOfWeek'] = df.index.dayofweek
#
#     # Adding cyclical features
#     df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
#     df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
#     df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
#     df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
#
#     # Adding holiday feature
#     cal = calendar()
#     holidays = cal.holidays(start=df.index.min(), end=df.index.max())
#     df['Holiday'] = df.index.isin(holidays).astype(int)
#
#     # Feature scaling
#     scaler = StandardScaler()
#     df['Vehicles_scaled'] = scaler.fit_transform(df[['Vehicles']])
#
#     return df
#
#
# # Initialize session state
# if 'uploaded_photos' not in st.session_state:
#     st.session_state.uploaded_photos = []
#
#
# # Main function to run the Streamlit app
# def main():
#     # Set page title and icon
#     st.set_page_config(page_title="Traffic Data Analysis", page_icon=":car:")
#
#     # Load data
#     df = load_data()
#
#     # Create sidebar navigation
#     st.sidebar.title("Navigation")
#     selected_tab = st.sidebar.radio("Go to", ["Traffic Data", "Upload Photos", "Settings"])
#
#     # Display selected tab content
#     if selected_tab == "Traffic Data":
#         display_traffic_data(df)
#     elif selected_tab == "Upload Photos":
#         display_upload_photos()
#     elif selected_tab == "Settings":
#         display_settings(df)
#
#
# # Function to display Traffic Data tab
# def display_traffic_data(df):
#     st.title("Traffic Data Analysis")
#
#     # Check if 'Year' column exists in the DataFrame
#     if 'Year' not in df.columns:
#         st.error("Year column not found in the DataFrame!")
#         return
#
#     # Initialize junctions variable
#     junctions = []
#
#     # Sidebar filters
#     st.sidebar.title("Traffic Data Filters")
#
#     # Checkbox for selecting specific date, year, and hour
#     show_date_checkbox = st.sidebar.checkbox("Filter by Date")
#
#     if show_date_checkbox:
#         st.sidebar.title("Date Filters")
#         year = st.sidebar.selectbox("Select Year", np.append(["Overall"], df['Year'].unique()))
#         month = st.sidebar.selectbox("Select Month", np.append(["Overall"], df['Month'].unique()))
#         day = st.sidebar.selectbox("Select Day", np.append(["Overall"], df['Day'].unique()))
#
#         if year == "Overall":
#             filtered_df = df
#         else:
#             filtered_df = df[df['Year'] == int(year)]
#
#         if month != "Overall":
#             filtered_df = filtered_df[filtered_df['Month'] == int(month)]
#
#         if day != "Overall":
#             filtered_df = filtered_df[filtered_df['Day'] == int(day)]
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}")
#
#     else:
#         st.sidebar.title("Date and Time Filters")
#         year = st.sidebar.selectbox("Select Year", np.append(["Overall"], df['Year'].unique()))
#         month = st.sidebar.selectbox("Select Month", np.append(["Overall"], df['Month'].unique()))
#         day = st.sidebar.selectbox("Select Day", np.append(["Overall"], df['Day'].unique()))
#         hour = st.sidebar.selectbox("Select Hour", np.append(["Overall"], df['Hour'].unique()))
#         junctions = st.sidebar.multiselect("Select Junction", df['Junction'].unique(), default=df['Junction'].unique())
#
#         filtered_df = df
#
#         if year != "Overall":
#             filtered_df = filtered_df[filtered_df['Year'] == int(year)]
#
#         if month != "Overall":
#             filtered_df = filtered_df[filtered_df['Month'] == int(month)]
#
#         if day != "Overall":
#             filtered_df = filtered_df[filtered_df['Day'] == int(day)]
#
#         if hour != "Overall":
#             filtered_df = filtered_df[filtered_df['Hour'] == int(hour)]
#
#         if junctions:
#             filtered_df = filtered_df[filtered_df['Junction'].isin(junctions)]
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}, Hour: {hour}, Junctions: {junctions}")
#
#     # Display filtered data
#     st.write(filtered_df)
#
#     # Visualization of filtered data
#     st.title("Historical Traffic Data - Line Chart")
#
#     if not filtered_df.empty:
#         # Group by Hour and Junction to get the average number of vehicles
#         historical_df = filtered_df.groupby(['Hour', 'Junction'], as_index=False)['Vehicles'].mean()
#
#         # Create line chart using Plotly Express
#         fig = px.line(historical_df, x='Hour', y='Vehicles', color='Junction',
#                       title='Average Hourly Traffic Data',
#                       labels={'Hour': 'Hour of Day', 'Vehicles': 'Average Number of Vehicles'})
#         st.plotly_chart(fig)
#
#
# # Function to display Upload Photos tab
# def display_upload_photos():
#     st.title("Upload Photos")
#
#     # Allow users to upload photos
#     uploaded_files = st.file_uploader("Choose a photo", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
#
#     # Display uploaded photos
#     if uploaded_files:
#         for uploaded_file in uploaded_files:
#             st.image(uploaded_file, caption=uploaded_file.name)
#
#             # Save uploaded file as bytes
#             img_bytes = uploaded_file.read()
#
#             # Add to session state for download
#             st.session_state.uploaded_photos.append({
#                 "name": uploaded_file.name,
#                 "data": img_bytes
#             })
#
#         # Display success message
#         st.success("Photos uploaded successfully!")
#
#     # Display option to download uploaded photos
#     if st.session_state.uploaded_photos:
#         st.title("Download Uploaded Photos")
#
#         # Display links for downloading uploaded photos
#         for i, photo_data in enumerate(st.session_state.uploaded_photos, start=1):
#             photo_name = photo_data["name"]
#             st.markdown(f"**Uploaded Photo {i}**: ")
#             st.markdown(get_image_download_link(photo_data["data"], photo_name), unsafe_allow_html=True)
#
#
# # Function to get download link for an image
# def get_image_download_link(img_bytes, text):
#     buffered = BytesIO(img_bytes)
#     img = Image.open(buffered)
#     img.save(buffered, format="JPEG")
#     img_str = base64.b64encode(buffered.getvalue()).decode()
#     href = f'<a href="data:file/jpg;base64,{img_str}" download="{text}.jpg">Download {text}</a>'
#     return href
#
#
# # Function to display Settings tab
# def display_settings(df):
#     st.title("Settings")
#
#     st.header("Visualization Settings")
#     # Example: Adjusting line chart granularity
#     line_chart_granularity = st.selectbox("Line Chart Granularity", ["Hour", "Day", "Month"])
#     st.write(f"Selected Line Chart Granularity: {line_chart_granularity}")
#
#     # Example: Adjusting color scheme
#     color_scheme = st.selectbox("Color Scheme", ["Plotly Express", "Streamlit Default", "Custom"])
#     st.write(f"Selected Color Scheme: {color_scheme}")
#
#     # Example: Adjusting plot size
#     plot_size = st.slider("Plot Size", min_value=200, max_value=1000, value=500, step=50)
#     st.write(f"Selected Plot Size: {plot_size}")
#
#     st.header("Data Filtering Defaults")
#     # Example: Default year selection
#     default_year = st.selectbox("Default Year", np.append(["Overall"], df['Year'].unique()))
#     st.write(f"Default Year: {default_year}")
#
#     # Example: Default junction selection
#     default_junction = st.multiselect("Default Junction", df['Junction'].unique(), default=df['Junction'].unique())
#     st.write(f"Default Junctions: {default_junction}")
#
#
# # Run the app
# if __name__ == "__main__":
#     main()

# import streamlit as st
# import pandas as pd
# import numpy as np
# from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
# from sklearn.preprocessing import StandardScaler
# import plotly.express as px
# from PIL import Image
# import base64
# from io import BytesIO  # Import BytesIO for handling bytes data
#
#
# # Load and preprocess data
# @st.cache
# def load_data():
#     # Load the data (replace with your data loading logic)
#     df = pd.read_csv("uncleaned_traffic.csv")
#
#     # Your data preprocessing steps go here
#     # Data cleaning
#     df = df.dropna(subset=['DateTime', 'Junction', 'Vehicles', 'ID'])
#     df = df.drop_duplicates()
#     df['DateTime'] = pd.to_datetime(df['DateTime'])
#     df['Vehicles'] = pd.to_numeric(df['Vehicles'], errors='coerce')  # Ensure 'Vehicles' column is numeric
#     df = df.dropna(subset=['Vehicles'])
#     df['Junction'] = df['Junction'].astype(int)
#     df['ID'] = df['ID'].apply(lambda x: '{:.0f}'.format(x))
#     df = df.drop_duplicates()
#     df = df.sort_values(by=['Junction', 'DateTime'])
#     df.reset_index(drop=True, inplace=True)
#     df.set_index('DateTime', inplace=True)
#
#     # Handling missing values
#     df.fillna(method='ffill', inplace=True)
#
#     # Extracting date-time features
#     df['Year'] = df.index.year
#     df['Month'] = df.index.month
#     df['Day'] = df.index.day
#     df['Hour'] = df.index.hour
#     df['DayOfWeek'] = df.index.dayofweek
#
#     # Adding cyclical features
#     df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
#     df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
#     df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
#     df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
#
#     # Adding holiday feature
#     cal = calendar()
#     holidays = cal.holidays(start=df.index.min(), end=df.index.max())
#     df['Holiday'] = df.index.isin(holidays).astype(int)
#
#     # Feature scaling
#     scaler = StandardScaler()
#     df['Vehicles_scaled'] = scaler.fit_transform(df[['Vehicles']])
#
#     return df
#
#
# # Initialize session state
# if 'uploaded_photos' not in st.session_state:
#     st.session_state.uploaded_photos = []
#
#
# # Main function to run the Streamlit app
# def main():
#     # Set page title and icon
#     st.set_page_config(page_title="Traffic Data Analysis", page_icon=":car:")
#
#     # Load data
#     df = load_data()
#
#     # Create sidebar navigation
#     st.sidebar.title("Navigation")
#     selected_tab = st.sidebar.radio("Go to", ["Traffic Data", "Upload Photos", "Settings"])
#
#     # Display selected tab content
#     if selected_tab == "Traffic Data":
#         display_traffic_data(df)
#     elif selected_tab == "Upload Photos":
#         display_upload_photos()
#     elif selected_tab == "Settings":
#         display_settings(df)
#
#
# # Function to display Traffic Data tab
# def display_traffic_data(df):
#     st.title("Traffic Data Analysis")
#
#     # Check if 'Year' column exists in the DataFrame
#     if 'Year' not in df.columns:
#         st.error("Year column not found in the DataFrame!")
#         return
#
#     # Initialize junctions variable
#     junctions = []
#
#     # Sidebar filters
#     st.sidebar.title("Traffic Data Filters")
#
#     # Checkbox for selecting specific date, year, and hour
#     show_date_checkbox = st.sidebar.checkbox("Filter by Date")
#
#     if show_date_checkbox:
#         st.sidebar.title("Date Filters")
#         year = st.sidebar.selectbox("Select Year", np.append(["Overall"], df['Year'].unique()))
#         month = st.sidebar.selectbox("Select Month", np.append(["Overall"], df['Month'].unique()))
#         day = st.sidebar.selectbox("Select Day", np.append(["Overall"], df['Day'].unique()))
#
#         if year == "Overall":
#             filtered_df = df
#         else:
#             filtered_df = df[df['Year'] == int(year)]
#
#         if month != "Overall":
#             filtered_df = filtered_df[filtered_df['Month'] == int(month)]
#
#         if day != "Overall":
#             filtered_df = filtered_df[filtered_df['Day'] == int(day)]
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}")
#
#     else:
#         st.sidebar.title("Date and Time Filters")
#         year = st.sidebar.selectbox("Select Year", np.append(["Overall"], df['Year'].unique()))
#         month = st.sidebar.selectbox("Select Month", np.append(["Overall"], df['Month'].unique()))
#         day = st.sidebar.selectbox("Select Day", np.append(["Overall"], df['Day'].unique()))
#         hour = st.sidebar.selectbox("Select Hour", np.append(["Overall"], df['Hour'].unique()))
#         junctions = st.sidebar.multiselect("Select Junction", df['Junction'].unique(), default=df['Junction'].unique())
#
#         filtered_df = df
#
#         if year != "Overall":
#             filtered_df = filtered_df[filtered_df['Year'] == int(year)]
#
#         if month != "Overall":
#             filtered_df = filtered_df[filtered_df['Month'] == int(month)]
#
#         if day != "Overall":
#             filtered_df = filtered_df[filtered_df['Day'] == int(day)]
#
#         if hour != "Overall":
#             filtered_df = filtered_df[filtered_df['Hour'] == int(hour)]
#
#         if junctions:
#             filtered_df = filtered_df[filtered_df['Junction'].isin(junctions)]
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}, Hour: {hour}, Junctions: {junctions}")
#
#     # Display filtered data
#     st.write(filtered_df)
#
#     # Visualization of filtered data
#     st.title("Historical Traffic Data - Line Chart")
#
#     if not filtered_df.empty:
#         # Group by Hour and Junction to get the average number of vehicles
#         historical_df = filtered_df.groupby(['Hour', 'Junction'], as_index=False)['Vehicles'].mean()
#
#         # Create line chart using Plotly Express
#         fig = px.line(historical_df, x='Hour', y='Vehicles', color='Junction',
#                       title='Average Hourly Traffic Data',
#                       labels={'Hour': 'Hour of Day', 'Vehicles': 'Average Number of Vehicles'})
#         st.plotly_chart(fig)
#
#
# # Function to display Upload Photos tab
# def display_upload_photos():
#     st.title("Upload Photos")
#
#     # Allow users to upload photos
#     uploaded_files = st.file_uploader("Choose a photo", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
#
#     # Display uploaded photos
#     if uploaded_files:
#         for uploaded_file in uploaded_files:
#             st.image(uploaded_file, caption=uploaded_file.name)
#
#             # Save uploaded file as bytes
#             img_bytes = uploaded_file.read()
#
#             # Add to session state for download
#             st.session_state.uploaded_photos.append({
#                 "name": uploaded_file.name,
#                 "data": img_bytes
#             })
#
#         # Display success message
#         st.success("Photos uploaded successfully!")
#
#     # Display option to download uploaded photos
#     if st.session_state.uploaded_photos:
#         st.title("Download Uploaded Photos")
#
#         # Display links for downloading uploaded photos
#         for i, photo_data in enumerate(st.session_state.uploaded_photos, start=1):
#             photo_name = photo_data.get("name", f"Uploaded Photo {i}")
#             st.markdown(f"**Uploaded Photo {i}**: ")
#             st.markdown(get_image_download_link(photo_data["data"], photo_name), unsafe_allow_html=True)
#
#
# # Function to get download link for an image
# def get_image_download_link(img_bytes, text):
#     buffered = BytesIO(img_bytes)
#     img = Image.open(buffered)
#     img.save(buffered, format="JPEG")
#     img_str = base64.b64encode(buffered.getvalue()).decode()
#     href = f'<a href="data:file/jpg;base64,{img_str}" download="{text}.jpg">Download {text}</a>'
#     return href
#
#
# # Function to display Settings tab
# def display_settings(df):
#     st.title("Settings")
#
#     st.header("Visualization Settings")
#     # Example: Adjusting line chart granularity
#     line_chart_granularity = st.selectbox("Line Chart Granularity", ["Hour", "Day", "Month"])
#     st.write(f"Selected Line Chart Granularity: {line_chart_granularity}")
#
#     # Example: Adjusting color scheme
#     color_scheme = st.selectbox("Color Scheme", ["Plotly Express", "Streamlit Default", "Custom"])
#     st.write(f"Selected Color Scheme: {color_scheme}")
#
#     # Example: Adjusting plot size
#     plot_size = st.slider("Plot Size", min_value=200, max_value=1000, value=500, step=50)
#     st.write(f"Selected Plot Size: {plot_size}")
#
#     st.header("Data Filtering Defaults")
#     # Example: Default year selection
#     default_year = st.selectbox("Default Year", np.append(["Overall"], df['Year'].unique()))
#     st.write(f"Default Year: {default_year}")
#
#     # Example: Default junction selection
#     default_junction = st.multiselect("Default Junction", df['Junction'].unique(), default=df['Junction'].unique())
#     st.write(f"Default Junctions: {default_junction}")
#
#
# # Run the app
# if __name__ == "__main__":
#     main()
#
# import streamlit as st
# import pandas as pd
# import numpy as np
# from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
# from sklearn.preprocessing import StandardScaler
# import plotly.express as px
# from PIL import Image
# import base64
# from io import BytesIO  # Import BytesIO for handling bytes data
#
#
# # Load and preprocess data
# @st.cache
# def load_data():
#     # Load the data (replace with your data loading logic)
#     df = pd.read_csv("uncleaned_traffic.csv")
#
#     # Your data preprocessing steps go here
#     # Data cleaning
#     df = df.dropna(subset=['DateTime', 'Junction', 'Vehicles', 'ID'])
#     df = df.drop_duplicates()
#     df['DateTime'] = pd.to_datetime(df['DateTime'])
#     df['Vehicles'] = pd.to_numeric(df['Vehicles'], errors='coerce')  # Ensure 'Vehicles' column is numeric
#     df = df.dropna(subset=['Vehicles'])
#     df['Junction'] = df['Junction'].astype(int)
#     df['ID'] = df['ID'].apply(lambda x: '{:.0f}'.format(x))
#     df = df.drop_duplicates()
#     df = df.sort_values(by=['Junction', 'DateTime'])
#     df.reset_index(drop=True, inplace=True)
#     df.set_index('DateTime', inplace=True)
#
#     # Handling missing values
#     df.fillna(method='ffill', inplace=True)
#
#     # Extracting date-time features
#     df['Year'] = df.index.year
#     df['Month'] = df.index.month
#     df['Day'] = df.index.day
#     df['Hour'] = df.index.hour
#     df['DayOfWeek'] = df.index.dayofweek
#
#     # Adding cyclical features
#     df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
#     df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
#     df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
#     df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
#
#     # Adding holiday feature
#     cal = calendar()
#     holidays = cal.holidays(start=df.index.min(), end=df.index.max())
#     df['Holiday'] = df.index.isin(holidays).astype(int)
#
#     # Feature scaling
#     scaler = StandardScaler()
#     df['Vehicles_scaled'] = scaler.fit_transform(df[['Vehicles']])
#
#     return df
#
#
# # Initialize session state
# if 'uploaded_photos' not in st.session_state:
#     st.session_state.uploaded_photos = []
#
#
# # Main function to run the Streamlit app
# def main():
#     # Set page title and icon
#     st.set_page_config(page_title="Traffic Data Analysis", page_icon=":car:")
#
#     # Load data
#     df = load_data()
#
#     # Create sidebar navigation
#     st.sidebar.title("Navigation")
#     selected_tab = st.sidebar.radio("Go to", ["Traffic Data", "Upload Photos", "Settings"])
#
#     # Display selected tab content
#     if selected_tab == "Traffic Data":
#         display_traffic_data(df)
#     elif selected_tab == "Upload Photos":
#         display_upload_photos()
#     elif selected_tab == "Settings":
#         display_settings(df)
#
#
# # Function to display Traffic Data tab
# def display_traffic_data(df):
#     st.title("Traffic Data Analysis")
#
#     # Check if 'Year' column exists in the DataFrame
#     if 'Year' not in df.columns:
#         st.error("Year column not found in the DataFrame!")
#         return
#
#     # Initialize junctions variable
#     junctions = []
#
#     # Sidebar filters
#     st.sidebar.title("Traffic Data Filters")
#
#     # Checkbox for selecting specific date, year, and hour
#     show_date_checkbox = st.sidebar.checkbox("Filter by Date")
#
#     if show_date_checkbox:
#         st.sidebar.title("Date Filters")
#         year = st.sidebar.selectbox("Select Year", np.append(["Overall"], df['Year'].unique()))
#         month = st.sidebar.selectbox("Select Month", np.append(["Overall"], df['Month'].unique()))
#         day = st.sidebar.selectbox("Select Day", np.append(["Overall"], df['Day'].unique()))
#
#         if year == "Overall":
#             filtered_df = df
#         else:
#             filtered_df = df[df['Year'] == int(year)]
#
#         if month != "Overall":
#             filtered_df = filtered_df[filtered_df['Month'] == int(month)]
#
#         if day != "Overall":
#             filtered_df = filtered_df[filtered_df['Day'] == int(day)]
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}")
#
#     else:
#         st.sidebar.title("Date and Time Filters")
#         year = st.sidebar.selectbox("Select Year", np.append(["Overall"], df['Year'].unique()))
#         month = st.sidebar.selectbox("Select Month", np.append(["Overall"], df['Month'].unique()))
#         day = st.sidebar.selectbox("Select Day", np.append(["Overall"], df['Day'].unique()))
#         hour = st.sidebar.selectbox("Select Hour", np.append(["Overall"], df['Hour'].unique()))
#         junctions = st.sidebar.multiselect("Select Junction", df['Junction'].unique(), default=df['Junction'].unique())
#
#         filtered_df = df
#
#         if year != "Overall":
#             filtered_df = filtered_df[filtered_df['Year'] == int(year)]
#
#         if month != "Overall":
#             filtered_df = filtered_df[filtered_df['Month'] == int(month)]
#
#         if day != "Overall":
#             filtered_df = filtered_df[filtered_df['Day'] == int(day)]
#
#         if hour != "Overall":
#             filtered_df = filtered_df[filtered_df['Hour'] == int(hour)]
#
#         if junctions:
#             filtered_df = filtered_df[filtered_df['Junction'].isin(junctions)]
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}, Hour: {hour}, Junctions: {junctions}")
#
#     # Display filtered data
#     st.write(filtered_df)
#
#     # Visualization of filtered data
#     st.title("Historical Traffic Data - Line Chart")
#
#     if not filtered_df.empty:
#         # Group by Hour and Junction to get the average number of vehicles
#         historical_df = filtered_df.groupby(['Hour', 'Junction'], as_index=False)['Vehicles'].mean()
#
#         # Create line chart using Plotly Express
#         fig = px.line(historical_df, x='Hour', y='Vehicles', color='Junction',
#                       title='Average Hourly Traffic Data',
#                       labels={'Hour': 'Hour of Day', 'Vehicles': 'Average Number of Vehicles'})
#         st.plotly_chart(fig)
#
#
# # Function to display Upload Photos tab
# def display_upload_photos():
#     st.title("Upload Photos")
#
#     # Allow users to upload photos
#     uploaded_files = st.file_uploader("Choose a photo", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
#
#     # Display uploaded photos
#     if uploaded_files:
#         for uploaded_file in uploaded_files:
#             st.image(uploaded_file, caption=uploaded_file.name)
#
#             # Save uploaded file as bytes
#             img_bytes = uploaded_file.read()
#
#             # Add to session state for download
#             st.session_state.uploaded_photos.append({
#                 "name": uploaded_file.name,
#                 "data": img_bytes
#             })
#
#         # Display success message
#         st.success("Photos uploaded successfully!")
#
#     # Display option to download uploaded photos
#     if st.session_state.uploaded_photos:
#         st.title("Download Uploaded Photos")
#
#         # Display links for downloading uploaded photos
#         for i, photo_data in enumerate(st.session_state.uploaded_photos, start=1):
#             photo_name = photo_data.get("name", f"Uploaded Photo {i}")
#             st.markdown(f"**Uploaded Photo {i}**: ")
#             st.markdown(get_image_download_link(photo_data["data"], photo_name), unsafe_allow_html=True)
#
#
# # Function to get download link for an image
# def get_image_download_link(img_bytes, text):
#     buffered = BytesIO(img_bytes)
#     img = Image.open(buffered)
#     img.save(buffered, format="JPEG")
#     img_str = base64.b64encode(buffered.getvalue()).decode()
#     href = f'<a href="data:file/jpg;base64,{img_str}" download="{text}.jpg">Download {text}</a>'
#     return href
#
#
# # Function to display Settings tab
# def display_settings(df):
#     st.title("Settings")
#
#     st.header("Visualization Settings")
#     # Example: Adjusting line chart granularity
#     line_chart_granularity = st.selectbox("Line Chart Granularity", ["Hour", "Day", "Month"])
#     st.write(f"Selected Line Chart Granularity: {line_chart_granularity}")
#
#     # Example: Adjusting color scheme
#     color_scheme = st.selectbox("Color Scheme", ["Plotly Express", "Streamlit Default", "Custom"])
#     st.write(f"Selected Color Scheme: {color_scheme}")
#
#     # Example: Adjusting plot size
#     plot_size = st.slider("Plot Size", min_value=200, max_value=1000, value=500, step=50)
#     st.write(f"Selected Plot Size: {plot_size}")
#
#     st.header("Data Filtering Defaults")
#     # Example: Default year selection
#     default_year = st.selectbox("Default Year", np.append(["Overall"], df['Year'].unique()))
#     st.write(f"Default Year: {default_year}")
#
#     # Example: Default junction selection
#     default_junction = st.multiselect("Default Junction", df['Junction'].unique(), default=df['Junction'].unique())
#     st.write(f"Default Junctions: {default_junction}")
#
#
# # Run the app
# if __name__ == "__main__":
#     main()

# import streamlit as st
# import pandas as pd
# import numpy as np
# from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
# from sklearn.preprocessing import StandardScaler
# import plotly.express as px
# from PIL import Image
# import base64
# from io import BytesIO  # Import BytesIO for handling bytes data
#
#
# # Load and preprocess data
# @st.cache_data
# def load_data():
#     # Load the data (replace with your data loading logic)
#     df = pd.read_csv("uncleaned_traffic.csv")
#
#     # Your data preprocessing steps go here
#     # Data cleaning
#     df = df.dropna(subset=['DateTime', 'Junction', 'Vehicles', 'ID'])
#     df = df.drop_duplicates()
#     df['DateTime'] = pd.to_datetime(df['DateTime'])
#     df['Vehicles'] = pd.to_numeric(df['Vehicles'], errors='coerce')  # Ensure 'Vehicles' column is numeric
#     df = df.dropna(subset=['Vehicles'])
#     df['Junction'] = df['Junction'].astype(int)
#     df['ID'] = df['ID'].apply(lambda x: '{:.0f}'.format(x))
#     df = df.drop_duplicates()
#     df = df.sort_values(by=['Junction', 'DateTime'])
#     df.reset_index(drop=True, inplace=True)
#     df.set_index('DateTime', inplace=True)
#
#     # Handling missing values
#     df.fillna(method='ffill', inplace=True)
#
#     # Extracting date-time features
#     df['Year'] = df.index.year
#     df['Month'] = df.index.month
#     df['Day'] = df.index.day
#     df['Hour'] = df.index.hour
#     df['DayOfWeek'] = df.index.dayofweek
#
#     # Adding cyclical features
#     df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
#     df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
#     df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
#     df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
#
#     # Adding holiday feature
#     cal = calendar()
#     holidays = cal.holidays(start=df.index.min(), end=df.index.max())
#     df['Holiday'] = df.index.isin(holidays).astype(int)
#
#     # Feature scaling
#     scaler = StandardScaler()
#     df['Vehicles_scaled'] = scaler.fit_transform(df[['Vehicles']])
#
#     return df
#
#
# # Initialize session state
# if 'uploaded_photos' not in st.session_state:
#     st.session_state.uploaded_photos = []
#
#
# # Main function to run the Streamlit app
# def main():
#     # Set page title and icon
#     st.set_page_config(page_title="Traffic Data Analysis", page_icon=":car:")
#
#     # Load data
#     df = load_data()
#
#     # Display title, logo, and traffic police picture
#     st.title("Traffic Data Analysis")
#     st.image("traffic-police.jpg", caption="Traffic Police", use_column_width=True)
#
#     # Create sidebar navigation
#     st.sidebar.title("Navigation")
#     selected_tab = st.sidebar.radio("Go to", ["Traffic Data", "Upload Photos", "Settings"])
#
#     # Display selected tab content
#     if selected_tab == "Traffic Data":
#         display_traffic_data(df)
#     elif selected_tab == "Upload Photos":
#         display_upload_photos()
#     elif selected_tab == "Settings":
#         display_settings(df)
#
#
# # Function to display Traffic Data tab
# def display_traffic_data(df):
#     st.title("Traffic Data Analysis")
#
#     # Check if 'Year' column exists in the DataFrame
#     if 'Year' not in df.columns:
#         st.error("Year column not found in the DataFrame!")
#         return
#
#     # Initialize junctions variable
#     junctions = []
#
#     # Sidebar filters
#     st.sidebar.title("Traffic Data Filters")
#
#     # Checkbox for selecting specific date, year, and hour
#     show_date_checkbox = st.sidebar.checkbox("Filter by Date")
#
#     if show_date_checkbox:
#         st.sidebar.title("Date Filters")
#         year = st.sidebar.selectbox("Select Year", np.append(["Overall"], df['Year'].unique()))
#         month = st.sidebar.selectbox("Select Month", np.append(["Overall"], df['Month'].unique()))
#         day = st.sidebar.selectbox("Select Day", np.append(["Overall"], df['Day'].unique()))
#
#         if year == "Overall":
#             filtered_df = df
#         else:
#             filtered_df = df[df['Year'] == int(year)]
#
#         if month != "Overall":
#             filtered_df = filtered_df[filtered_df['Month'] == int(month)]
#
#         if day != "Overall":
#             filtered_df = filtered_df[filtered_df['Day'] == int(day)]
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}")
#
#     else:
#         st.sidebar.title("Date and Time Filters")
#         year = st.sidebar.selectbox("Select Year", np.append(["Overall"], df['Year'].unique()))
#         month = st.sidebar.selectbox("Select Month", np.append(["Overall"], df['Month'].unique()))
#         day = st.sidebar.selectbox("Select Day", np.append(["Overall"], df['Day'].unique()))
#         hour = st.sidebar.selectbox("Select Hour", np.append(["Overall"], df['Hour'].unique()))
#         junctions = st.sidebar.multiselect("Select Junction", df['Junction'].unique(), default=df['Junction'].unique())
#
#         filtered_df = df
#
#         if year != "Overall":
#             filtered_df = filtered_df[filtered_df['Year'] == int(year)]
#
#         if month != "Overall":
#             filtered_df = filtered_df[filtered_df['Month'] == int(month)]
#
#         if day != "Overall":
#             filtered_df = filtered_df[filtered_df['Day'] == int(day)]
#
#         if hour != "Overall":
#             filtered_df = filtered_df[filtered_df['Hour'] == int(hour)]
#
#         if junctions:
#             filtered_df = filtered_df[filtered_df['Junction'].isin(junctions)]
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}, Hour: {hour}, Junctions: {junctions}")
#
#     # Display filtered data
#     st.write(filtered_df)
#
#     # Visualization of filtered data
#     st.title("Historical Traffic Data - Line Chart")
#
#     if not filtered_df.empty:
#         # Group by Hour and Junction to get the average number of vehicles
#         historical_df = filtered_df.groupby(['Hour', 'Junction'], as_index=False)['Vehicles'].mean()
#
#         # Create line chart using Plotly Express
#         fig = px.line(historical_df, x='Hour', y='Vehicles', color='Junction',
#                       title='Average Hourly Traffic Data',
#                       labels={'Hour': 'Hour of Day', 'Vehicles': 'Average Number of Vehicles'})
#         st.plotly_chart(fig)
#
#
# # Function to display Upload Photos tab
# def display_upload_photos():
#     st.title("Upload Photos")
#
#     # Allow users to upload photos
#     uploaded_files = st.file_uploader("Choose a photo", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
#
#     # Display uploaded photos
#     if uploaded_files:
#         for uploaded_file in uploaded_files:
#             st.image(uploaded_file, caption=uploaded_file.name)
#
#             # Save uploaded file as bytes
#             img_bytes = uploaded_file.read()
#
#             # Add to session state for download
#             st.session_state.uploaded_photos.append({
#                 "name": uploaded_file.name,
#                 "data": img_bytes
#             })
#
#         # Display success message
#         st.success("Photos uploaded successfully!")
#
#     # Display option to download uploaded photos
#     if st.session_state.uploaded_photos:
#         st.title("Download Uploaded Photos")
#
#         # Display links for downloading uploaded photos
#         for i, photo_data in enumerate(st.session_state.uploaded_photos, start=1):
#             photo_name = photo_data.get("name", f"Uploaded Photo {i}")
#             st.markdown(f"**Uploaded Photo {i}**: ")
#             st.markdown(get_image_download_link(photo_data["data"], photo_name), unsafe_allow_html=True)
#
#
# # Function to get download link for an image
# def get_image_download_link(img_bytes, text):
#     buffered = BytesIO(img_bytes)
#     img = Image.open(buffered)
#     img.save(buffered, format="JPEG")
#     img_str = base64.b64encode(buffered.getvalue()).decode()
#     href = f'<a href="data:file/jpg;base64,{img_str}" download="{text}.jpg">Download {text}</a>'
#     return href
#
#
# # Function to display Settings tab
# def display_settings(df):
#     st.title("Settings")
#
#     st.header("Visualization Settings")
#     # Example: Adjusting line chart granularity
#     line_chart_granularity = st.selectbox("Line Chart Granularity", ["Hour", "Day", "Month"])
#     st.write(f"Selected Line Chart Granularity: {line_chart_granularity}")
#
#     # Example: Adjusting color scheme
#     color_scheme = st.selectbox("Color Scheme", ["Plotly Express", "Streamlit Default", "Custom"])
#     st.write(f"Selected Color Scheme: {color_scheme}")
#
#     # Example: Adjusting plot size
#     plot_size = st.slider("Plot Size", min_value=200, max_value=1000, value=500, step=50)
#     st.write(f"Selected Plot Size: {plot_size}")
#
#     st.header("Data Filtering Defaults")
#     # Example: Default year selection
#     default_year = st.selectbox("Default Year", np.append(["Overall"], df['Year'].unique()))
#     st.write(f"Default Year: {default_year}")
#
#     # Example: Default junction selection
#     default_junction = st.multiselect("Default Junction", df['Junction'].unique(), default=df['Junction'].unique())
#     st.write(f"Default Junctions: {default_junction}")
#
#
# # Run the app
# if __name__ == "__main__":
#     main()

# import streamlit as st
# import pandas as pd
# import numpy as np
# from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
# from sklearn.preprocessing import StandardScaler
# import plotly.express as px
# from PIL import Image
# import base64
# from io import BytesIO  # Import BytesIO for handling bytes data
#
#
# # Load and preprocess data
# @st.cache_data
# def load_data():
#     # Load the data (replace with your data loading logic)
#     df = pd.read_csv("uncleaned_traffic.csv")
#
#     # Your data preprocessing steps go here
#     # Data cleaning
#     df = df.dropna(subset=['DateTime', 'Junction', 'Vehicles', 'ID'])
#     df = df.drop_duplicates()
#     df['DateTime'] = pd.to_datetime(df['DateTime'])
#     df['Vehicles'] = pd.to_numeric(df['Vehicles'], errors='coerce')  # Ensure 'Vehicles' column is numeric
#     df = df.dropna(subset=['Vehicles'])
#     df['Junction'] = df['Junction'].astype(int)
#     df['ID'] = df['ID'].apply(lambda x: '{:.0f}'.format(x))
#     df = df.drop_duplicates()
#     df = df.sort_values(by=['Junction', 'DateTime'])
#     df.reset_index(drop=True, inplace=True)
#     df.set_index('DateTime', inplace=True)
#
#     # Handling missing values
#     df.fillna(method='ffill', inplace=True)
#
#     # Extracting date-time features
#     df['Year'] = df.index.year
#     df['Month'] = df.index.month
#     df['Day'] = df.index.day
#     df['Hour'] = df.index.hour
#     df['DayOfWeek'] = df.index.dayofweek
#
#     # Adding cyclical features
#     df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
#     df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
#     df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
#     df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
#
#     # Adding holiday feature
#     cal = calendar()
#     holidays = cal.holidays(start=df.index.min(), end=df.index.max())
#     df['Holiday'] = df.index.isin(holidays).astype(int)
#
#     # Feature scaling
#     scaler = StandardScaler()
#     df['Vehicles_scaled'] = scaler.fit_transform(df[['Vehicles']])
#
#     return df
#
#
# # Initialize session state
# if 'uploaded_photos' not in st.session_state:
#     st.session_state.uploaded_photos = []
#
#
# # Main function to run the Streamlit app
# def main():
#     # Set page title and icon
#     st.set_page_config(page_title=" Welcome to Traffic Analyzer", page_icon=":car:")
#
#     # Load data
#     df = load_data()
#
#     # Display title, logo, and traffic police picture
#     st.title("Traffic Analyzer")
#     st.image("traffic-police.jpg", caption="Traffic Police", use_column_width=True)
#
#     # Logo on the left side
#     col1, col2 = st.beta_columns([1, 3])
#     with col1:
#         st.image("logo.jpg", width=100)
#
#     # Logo size on the right side
#     with col2:
#         st.image("right_logo.png", width=100)
#
#     # Create sidebar navigation
#     st.sidebar.title("Navigation")
#     selected_tab = st.sidebar.radio("Go to", ["Traffic Data", "Upload Photos", "Settings"])
#
#     # Display selected tab content
#     if selected_tab == "Traffic Data":
#         display_traffic_data(df)
#     elif selected_tab == "Upload Photos":
#         display_upload_photos()
#     elif selected_tab == "Settings":
#         display_settings(df)
#
#
# # Function to display Traffic Data tab
# def display_traffic_data(df):
#     st.title(" Traffic Data ")
#
#     # Check if 'Year' column exists in the DataFrame
#     if 'Year' not in df.columns:
#         st.error("Year column not found in the DataFrame!")
#         return
#
#     # Initialize junctions variable
#     junctions = []
#
#     # Sidebar filters
#     st.sidebar.title("Traffic Data Filters")
#
#     # Checkbox for selecting specific date, year, and hour
#     show_date_checkbox = st.sidebar.checkbox("Filter by Date")
#
#     if show_date_checkbox:
#         st.sidebar.title("Date Filters")
#         year = st.sidebar.selectbox("Select Year", np.append(["Overall"], df['Year'].unique()))
#         month = st.sidebar.selectbox("Select Month", np.append(["Overall"], df['Month'].unique()))
#         day = st.sidebar.selectbox("Select Day", np.append(["Overall"], df['Day'].unique()))
#
#         if year == "Overall":
#             filtered_df = df
#         else:
#             filtered_df = df[df['Year'] == int(year)]
#
#         if month != "Overall":
#             filtered_df = filtered_df[filtered_df['Month'] == int(month)]
#
#         if day != "Overall":
#             filtered_df = filtered_df[filtered_df['Day'] == int(day)]
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}")
#
#     else:
#         st.sidebar.title("Date and Time Filters")
#         year = st.sidebar.selectbox("Select Year", np.append(["Overall"], df['Year'].unique()))
#         month = st.sidebar.selectbox("Select Month", np.append(["Overall"], df['Month'].unique()))
#         day = st.sidebar.selectbox("Select Day", np.append(["Overall"], df['Day'].unique()))
#         hour = st.sidebar.selectbox("Select Hour", np.append(["Overall"], df['Hour'].unique()))
#         junctions = st.sidebar.multiselect("Select Junction", df['Junction'].unique(), default=df['Junction'].unique())
#
#         filtered_df = df
#
#         if year != "Overall":
#             filtered_df = filtered_df[filtered_df['Year'] == int(year)]
#
#         if month != "Overall":
#             filtered_df = filtered_df[filtered_df['Month'] == int(month)]
#
#         if day != "Overall":
#             filtered_df = filtered_df[filtered_df['Day'] == int(day)]
#
#         if hour != "Overall":
#             filtered_df = filtered_df[filtered_df['Hour'] == int(hour)]
#
#         if junctions:
#             filtered_df = filtered_df[filtered_df['Junction'].isin(junctions)]
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}, Hour: {hour}, Junctions: {junctions}")
#
#     # Display filtered data
#     st.write(filtered_df)
#
#     # Visualization of filtered data
#     st.title("Historical Traffic Data - Line Chart")
#
#     if not filtered_df.empty:
#         # Group by Hour and Junction to get the average number of vehicles
#         historical_df = filtered_df.groupby(['Hour', 'Junction'], as_index=False)['Vehicles'].mean()
#
#         # Create line chart using Plotly Express
#         fig = px.line(historical_df, x='Hour', y='Vehicles', color='Junction',
#                       title='Average Hourly Traffic Data',
#                       labels={'Hour': 'Hour of Day', 'Vehicles': 'Average Number of Vehicles'})
#         st.plotly_chart(fig)
#
#
# # Function to display Upload Photos tab
# def display_upload_photos():
#     st.title("Upload Photos")
#
#     # Allow users to upload photos
#     uploaded_files = st.file_uploader("Choose a photo", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
#
#     # Display uploaded photos
#     if uploaded_files:
#         for uploaded_file in uploaded_files:
#             st.image(uploaded_file, caption=uploaded_file.name)
#
#             # Save uploaded file as bytes
#             img_bytes = uploaded_file.read()
#
#             # Add to session state for download
#             st.session_state.uploaded_photos.append({
#                 "name": uploaded_file.name,
#                 "data": img_bytes
#             })
#
#         # Display success message
#         st.success("Photos uploaded successfully!")
#
#     # Display option to download uploaded photos
#     if st.session_state.uploaded_photos:
#         st.title("Download Uploaded Photos")
#
#         # Display links for downloading uploaded photos
#         for i, photo_data in enumerate(st.session_state.uploaded_photos, start=1):
#             photo_name = photo_data.get("name", f"Uploaded Photo {i}")
#             st.markdown(f"**Uploaded Photo {i}**: ")
#             st.markdown(get_image_download_link(photo_data["data"], photo_name), unsafe_allow_html=True)
#
#
# # Function to get download link for an image
# def get_image_download_link(img_bytes, text):
#     buffered = BytesIO(img_bytes)
#     img = Image.open(buffered)
#     img.save(buffered, format="JPEG")
#     img_str = base64.b64encode(buffered.getvalue()).decode()
#     href = f'<a href="data:file/jpg;base64,{img_str}" download="{text}.jpg">Download {text}</a>'
#     return href
#
#
# # Function to display Settings tab
# def display_settings(df):
#     st.title("Settings")
#
#     st.header("Visualization Settings")
#     # Example: Adjusting line chart granularity
#     line_chart_granularity = st.selectbox("Line Chart Granularity", ["Hour", "Day", "Month"])
#     st.write(f"Selected Line Chart Granularity: {line_chart_granularity}")
#
#     # Example: Adjusting color scheme
#     color_scheme = st.selectbox("Color Scheme", ["Plotly Express", "Streamlit Default", "Custom"])
#     st.write(f"Selected Color Scheme: {color_scheme}")
#
#     # Example: Adjusting plot size
#     plot_size = st.slider("Plot Size", min_value=200, max_value=1000, value=500, step=50)
#     st.write(f"Selected Plot Size: {plot_size}")
#
#     st.header("Data Filtering Defaults")
#     # Example: Default year selection
#     default_year = st.selectbox("Default Year", np.append(["Overall"], df['Year'].unique()))
#     st.write(f"Default Year: {default_year}")
#
#     # Example: Default junction selection
#     default_junction = st.multiselect("Default Junction", df['Junction'].unique(), default=df['Junction'].unique())
#     st.write(f"Default Junctions: {default_junction}")
#
#
# # Run the app
# if __name__ == "__main__":
#     main()

# import streamlit as st
# import pandas as pd
# import numpy as np
# from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
# from sklearn.preprocessing import StandardScaler
# import plotly.express as px
# from PIL import Image
# import base64
# from io import BytesIO  # Import BytesIO for handling bytes data
#
#
# # Load and preprocess data
# @st.cache_data
# def load_data():
#     # Load the data (replace with your data loading logic)
#     df = pd.read_csv("uncleaned_traffic.csv")
#
#     # Your data preprocessing steps go here
#     # Data cleaning
#     df = df.dropna(subset=['DateTime', 'Junction', 'Vehicles', 'ID'])
#     df = df.drop_duplicates()
#     df['DateTime'] = pd.to_datetime(df['DateTime'])
#     df['Vehicles'] = pd.to_numeric(df['Vehicles'], errors='coerce')  # Ensure 'Vehicles' column is numeric
#     df = df.dropna(subset=['Vehicles'])
#     df['Junction'] = df['Junction'].astype(int)
#     df['ID'] = df['ID'].apply(lambda x: '{:.0f}'.format(x))
#     df = df.drop_duplicates()
#     df = df.sort_values(by=['Junction', 'DateTime'])
#     df.reset_index(drop=True, inplace=True)
#     df.set_index('DateTime', inplace=True)
#
#     # Handling missing values
#     df.fillna(method='ffill', inplace=True)
#
#     # Extracting date-time features
#     df['Year'] = df.index.year
#     df['Month'] = df.index.month
#     df['Day'] = df.index.day
#     df['Hour'] = df.index.hour
#     df['DayOfWeek'] = df.index.dayofweek
#
#     # Adding cyclical features
#     df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
#     df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
#     df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
#     df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
#
#     # Adding holiday feature
#     cal = calendar()
#     holidays = cal.holidays(start=df.index.min(), end=df.index.max())
#     df['Holiday'] = df.index.isin(holidays).astype(int)
#
#     # Feature scaling
#     scaler = StandardScaler()
#     df['Vehicles_scaled'] = scaler.fit_transform(df[['Vehicles']])
#
#     return df
#
#
# # Initialize session state
# if 'uploaded_photos' not in st.session_state:
#     st.session_state.uploaded_photos = []
#
#
# # Main function to run the Streamlit app
# def main():
#     # Set page title and icon
#     st.set_page_config(page_title="Traffic Data Analysis", page_icon=":car:")
#
#     # Load data
#     df = load_data()
#
#     # Create header with title and logos
#     st.title("Traffic Data Analysis")
#
#     # Display logos on header
#     col1, col2 = st.beta_columns([1, 3])
#     with col1:
#         st.image("traffic-police.jpg", width=100, caption="Left Logo")
#
#     with col2:
#         st.image("logo.jpg", width=100, caption="Right Logo")
#
#     # Create sidebar navigation
#     st.sidebar.title("Navigation")
#     selected_tab = st.sidebar.radio("Go to", ["Traffic Data", "Upload Photos", "Settings"])
#
#     # Display selected tab content
#     if selected_tab == "Traffic Data":
#         display_traffic_data(df)
#     elif selected_tab == "Upload Photos":
#         display_upload_photos()
#     elif selected_tab == "Settings":
#         display_settings(df)
#
#
# # Function to display Traffic Data tab
# def display_traffic_data(df):
#     st.title("Traffic Data Analysis")
#
#     # Check if 'Year' column exists in the DataFrame
#     if 'Year' not in df.columns:
#         st.error("Year column not found in the DataFrame!")
#         return
#
#     # Initialize junctions variable
#     junctions = []
#
#     # Sidebar filters
#     st.sidebar.title("Traffic Data Filters")
#
#     # Checkbox for selecting specific date, year, and hour
#     show_date_checkbox = st.sidebar.checkbox("Filter by Date")
#
#     if show_date_checkbox:
#         st.sidebar.title("Date Filters")
#         year = st.sidebar.selectbox("Select Year", np.append(["Overall"], df['Year'].unique()))
#         month = st.sidebar.selectbox("Select Month", np.append(["Overall"], df['Month'].unique()))
#         day = st.sidebar.selectbox("Select Day", np.append(["Overall"], df['Day'].unique()))
#
#         if year == "Overall":
#             filtered_df = df
#         else:
#             filtered_df = df[df['Year'] == int(year)]
#
#         if month != "Overall":
#             filtered_df = filtered_df[filtered_df['Month'] == int(month)]
#
#         if day != "Overall":
#             filtered_df = filtered_df[filtered_df['Day'] == int(day)]
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}")
#
#     else:
#         st.sidebar.title("Date and Time Filters")
#         year = st.sidebar.selectbox("Select Year", np.append(["Overall"], df['Year'].unique()))
#         month = st.sidebar.selectbox("Select Month", np.append(["Overall"], df['Month'].unique()))
#         day = st.sidebar.selectbox("Select Day", np.append(["Overall"], df['Day'].unique()))
#         hour = st.sidebar.selectbox("Select Hour", np.append(["Overall"], df['Hour'].unique()))
#         junctions = st.sidebar.multiselect("Select Junction", df['Junction'].unique(), default=df['Junction'].unique())
#
#         filtered_df = df
#
#         if year != "Overall":
#             filtered_df = filtered_df[filtered_df['Year'] == int(year)]
#
#         if month != "Overall":
#             filtered_df = filtered_df[filtered_df['Month'] == int(month)]
#
#         if day != "Overall":
#             filtered_df = filtered_df[filtered_df['Day'] == int(day)]
#
#         if hour != "Overall":
#             filtered_df = filtered_df[filtered_df['Hour'] == int(hour)]
#
#         if junctions:
#             filtered_df = filtered_df[filtered_df['Junction'].isin(junctions)]
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}, Hour: {hour}, Junctions: {junctions}")
#
#     # Display filtered data
#     st.write(filtered_df)
#
#     # Visualization of filtered data
#     st.title("Historical Traffic Data - Line Chart")
#
#     if not filtered_df.empty:
#         # Group by Hour and Junction to get the average number of vehicles
#         historical_df = filtered_df.groupby(['Hour', 'Junction'], as_index=False)['Vehicles'].mean()
#
#         # Create line chart using Plotly Express
#         fig = px.line(historical_df, x='Hour', y='Vehicles', color='Junction',
#                       title='Average Hourly Traffic Data',
#                       labels={'Hour': 'Hour of Day', 'Vehicles': 'Average Number of Vehicles'})
#         st.plotly_chart(fig)
#
#
# # Function to display Upload Photos tab
# def display_upload_photos():
#     st.title("Upload Photos")
#
#     # Allow users to upload photos
#     uploaded_files = st.file_uploader("Choose a photo", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
#
#     # Display uploaded photos
#     if uploaded_files:
#         for uploaded_file in uploaded_files:
#             # Resize the image to a smaller size for display
#             img = Image.open(uploaded_file)
#             img = img.resize((200, 200))  # Adjust the size as needed
#             st.image(img, caption=uploaded_file.name)
#
#         # Append uploaded files to session state
#         st.session_state.uploaded_photos.extend(uploaded_files)
#
#         # Display success message
#         st.success("Photos uploaded successfully!")
#
#     # Display option to download uploaded photos
#     if st.session_state.uploaded_photos:
#         st.title("Download Uploaded Photos")
#
#         # Display links for downloading uploaded photos
#         for i, photo_data in enumerate(st.session_state.uploaded_photos, start=1):
#             photo_name = photo_data.get("name", f"Uploaded Photo {i}")
#             st.markdown(f"**{photo_name}**: ")
#             st.markdown(get_image_download_link(photo_data["data"], photo_name), unsafe_allow_html=True)
#
#
# # Function to get download link for an image
# def get_image_download_link(img, text):
#     buffered = BytesIO()
#     img.save(buffered, format="JPEG")
#     img_str = base64.b64encode(buffered.getvalue()).decode()
#     href = f'<a href="data:file/jpg;base64,{img_str}" download="{text}.jpg">Download {text}</a>'
#     return href
#
#
# # Function to display Settings tab
# def display_settings(df):
#     st.title("Settings")
#
#     st.header("Visualization Settings")
#     # Example: Adjusting line chart granularity
#     line_chart_granularity = st.selectbox("Line Chart Granularity", ["Hour", "Day", "Month"])
#     st.write(f"Selected Line Chart Granularity: {line_chart_granularity}")
#
#     # Example: Adjusting color scheme
#     color_scheme = st.selectbox("Color Scheme", ["Plotly Express", "Streamlit Default", "Custom"])
#     st.write(f"Selected Color Scheme: {color_scheme}")
#
#     # Example: Adjusting plot size
#     plot_size = st.slider("Plot Size", min_value=200, max_value=1000, value=500, step=50)
#     st.write(f"Selected Plot Size: {plot_size}")
#
#     st.header("Data Filtering Defaults")
#     # Example: Default year selection
#     default_year = st.selectbox("Default Year", df['Year'].unique())
#     st.write(f"Selected Default Year: {default_year}")
#
#
# # Run the main function
# if __name__ == "__main__":
#     main()

# import streamlit as st
# import pandas as pd
# import numpy as np
# from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
# from sklearn.preprocessing import StandardScaler
# import plotly.express as px
# from PIL import Image
# import base64
# from io import BytesIO  # Import BytesIO for handling bytes data
#
#
# # Load and preprocess data
# @st.cache_data
# def load_data():
#     # Load the data (replace with your data loading logic)
#     df = pd.read_csv("uncleaned_traffic.csv")
#
#     # Your data preprocessing steps go here
#     # Data cleaning
#     df = df.dropna(subset=['DateTime', 'Junction', 'Vehicles', 'ID'])
#     df = df.drop_duplicates()
#     df['DateTime'] = pd.to_datetime(df['DateTime'])
#     df['Vehicles'] = pd.to_numeric(df['Vehicles'], errors='coerce')  # Ensure 'Vehicles' column is numeric
#     df = df.dropna(subset=['Vehicles'])
#     df['Junction'] = df['Junction'].astype(int)
#     df['ID'] = df['ID'].apply(lambda x: '{:.0f}'.format(x))
#     df = df.drop_duplicates()
#     df = df.sort_values(by=['Junction', 'DateTime'])
#     df.reset_index(drop=True, inplace=True)
#     df.set_index('DateTime', inplace=True)
#
#     # Handling missing values
#     df.fillna(method='ffill', inplace=True)
#
#     # Extracting date-time features
#     df['Year'] = df.index.year
#     df['Month'] = df.index.month
#     df['Day'] = df.index.day
#     df['Hour'] = df.index.hour
#     df['DayOfWeek'] = df.index.dayofweek
#
#     # Adding cyclical features
#     df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
#     df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
#     df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
#     df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
#
#     # Adding holiday feature
#     cal = calendar()
#     holidays = cal.holidays(start=df.index.min(), end=df.index.max())
#     df['Holiday'] = df.index.isin(holidays).astype(int)
#
#     # Feature scaling
#     scaler = StandardScaler()
#     df['Vehicles_scaled'] = scaler.fit_transform(df[['Vehicles']])
#
#     return df
#
#
# # Initialize session state
# if 'uploaded_photos' not in st.session_state:
#     st.session_state.uploaded_photos = []
#
#
# # Main function to run the Streamlit app
# def main():
#     # Set page title and icon
#     st.set_page_config(page_title="Traffic Data Analysis", page_icon=":car:")
#
#     # Load data
#     df = load_data()
#
#     # Create header with title and logos
#     st.title("Traffic Data Analysis")
#
#     # Display logos on header using columns
#     col1, col2 = st.columns([1, 3])
#     with col1:
#         st.image("traffic-police.jpg", width=100, caption="Left Logo")
#
#     with col2:
#         st.image("logo.jpg", width=100, caption="Right Logo")
#
#     # Create sidebar navigation
#     st.sidebar.title("Navigation")
#     selected_tab = st.sidebar.radio("Go to", ["Traffic Data", "Upload Photos", "Settings"])
#
#     # Display selected tab content
#     if selected_tab == "Traffic Data":
#         display_traffic_data(df)
#     elif selected_tab == "Upload Photos":
#         display_upload_photos()
#     elif selected_tab == "Settings":
#         display_settings(df)
#
#
# # Function to display Traffic Data tab
# def display_traffic_data(df):
#     st.title("Traffic Data Analysis")
#
#     # Check if 'Year' column exists in the DataFrame
#     if 'Year' not in df.columns:
#         st.error("Year column not found in the DataFrame!")
#         return
#
#     # Initialize junctions variable
#     junctions = []
#
#     # Sidebar filters
#     st.sidebar.title("Traffic Data Filters")
#
#     # Checkbox for selecting specific date, year, and hour
#     show_date_checkbox = st.sidebar.checkbox("Filter by Date")
#
#     if show_date_checkbox:
#         st.sidebar.title("Date Filters")
#         year = st.sidebar.selectbox("Select Year", np.append(["Overall"], df['Year'].unique()))
#         month = st.sidebar.selectbox("Select Month", np.append(["Overall"], df['Month'].unique()))
#         day = st.sidebar.selectbox("Select Day", np.append(["Overall"], df['Day'].unique()))
#
#         if year == "Overall":
#             filtered_df = df
#         else:
#             filtered_df = df[df['Year'] == int(year)]
#
#         if month != "Overall":
#             filtered_df = filtered_df[filtered_df['Month'] == int(month)]
#
#         if day != "Overall":
#             filtered_df = filtered_df[filtered_df['Day'] == int(day)]
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}")
#
#     else:
#         st.sidebar.title("Date and Time Filters")
#         year = st.sidebar.selectbox("Select Year", np.append(["Overall"], df['Year'].unique()))
#         month = st.sidebar.selectbox("Select Month", np.append(["Overall"], df['Month'].unique()))
#         day = st.sidebar.selectbox("Select Day", np.append(["Overall"], df['Day'].unique()))
#         hour = st.sidebar.selectbox("Select Hour", np.append(["Overall"], df['Hour'].unique()))
#         junctions = st.sidebar.multiselect("Select Junction", df['Junction'].unique(), default=df['Junction'].unique())
#
#         filtered_df = df
#
#         if year != "Overall":
#             filtered_df = filtered_df[filtered_df['Year'] == int(year)]
#
#         if month != "Overall":
#             filtered_df = filtered_df[filtered_df['Month'] == int(month)]
#
#         if day != "Overall":
#             filtered_df = filtered_df[filtered_df['Day'] == int(day)]
#
#         if hour != "Overall":
#             filtered_df = filtered_df[filtered_df['Hour'] == int(hour)]
#
#         if junctions:
#             filtered_df = filtered_df[filtered_df['Junction'].isin(junctions)]
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}, Hour: {hour}, Junctions: {junctions}")
#
#     # Display filtered data
#     st.write(filtered_df)
#
#     # Visualization of filtered data
#     st.title("Historical Traffic Data - Line Chart")
#
#     if not filtered_df.empty:
#         # Group by Hour and Junction to get the average number of vehicles
#         historical_df = filtered_df.groupby(['Hour', 'Junction'], as_index=False)['Vehicles'].mean()
#
#         # Create line chart using Plotly Express
#         fig = px.line(historical_df, x='Hour', y='Vehicles', color='Junction',
#                       title='Average Hourly Traffic Data',
#                       labels={'Hour': 'Hour of Day', 'Vehicles': 'Average Number of Vehicles'})
#         st.plotly_chart(fig)
#
#
# # Function to display Upload Photos tab
# def display_upload_photos():
#     st.title("Upload Photos")
#
#     # Allow users to upload photos
#     uploaded_files = st.file_uploader("Choose a photo", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
#
#     # Display uploaded photos
#     if uploaded_files:
#         for uploaded_file in uploaded_files:
#             # Resize the image to a smaller size for display
#             img = Image.open(uploaded_file)
#             img = img.resize((200, 200))  # Adjust the size as needed
#             st.image(img, caption=uploaded_file.name)
#
#         # Append uploaded files to session state
#         st.session_state.uploaded_photos.extend(uploaded_files)
#
#         # Display success message
#         st.success("Photos uploaded successfully!")
#
#     # Display option to download uploaded photos
#     if st.session_state.uploaded_photos:
#         st.title("Download Uploaded Photos")
#
#         # Display links for downloading uploaded photos
#         for i, photo_data in enumerate(st.session_state.uploaded_photos, start=1):
#             photo_name = photo_data.get("name", f"Uploaded Photo {i}")
#             st.markdown(f"**{photo_name}**: ")
#             st.markdown(get_image_download_link(photo_data["data"], photo_name), unsafe_allow_html=True)
#
#
# # Function to get download link for an image
# def get_image_download_link(img, text):
#     buffered = BytesIO()
#     img.save(buffered, format="JPEG")
#     img_str = base64.b64encode(buffered.getvalue()).decode()
#     href = f'<a href="data:file/jpg;base64,{img_str}" download="{text}.jpg">Download {text}</a>'
#     return href
#
#
# # Function to display Settings tab
# def display_settings(df):
#     st.title("Settings")
#
#     st.header("Visualization Settings")
#     # Example: Adjusting line chart granularity
#     line_chart_granularity = st.selectbox("Line Chart Granularity", ["Hour", "Day", "Month"])
#     st.write(f"Selected Line Chart Granularity: {line_chart_granularity}")
#
#     # Example: Adjusting color scheme
#     color_scheme = st.selectbox("Color Scheme", ["Plotly Express", "Streamlit Default", "Custom"])
#     st.write(f"Selected Color Scheme: {color_scheme}")
#
#     # Example: Adjusting plot size
#     plot_size = st.slider("Plot Size", min_value=200, max_value=1000, value=500, step=50)
#     st.write(f"Selected Plot Size: {plot_size}")
#
#     st.header("Data Filtering Defaults")
#     # Example: Default year selection
#     default_year = st.selectbox("Default Year", df['Year'].unique())
#     st.write(f"Selected Default Year: {default_year}")
#
#
# # Run the main function
# if __name__ == "__main__":
#     main()

# import streamlit as st
# import pandas as pd
# import numpy as np
# from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
# from sklearn.preprocessing import StandardScaler
# import plotly.express as px
# from PIL import Image
# import base64
# from io import BytesIO  # Import BytesIO for handling bytes data
#
# # Load and preprocess data
# @st.cache_data
# def load_data():
#     # Load the data (replace with your data loading logic)
#     df = pd.read_csv("uncleaned_traffic.csv")
#
#     # Your data preprocessing steps go here
#     # Data cleaning
#     df = df.dropna(subset=['DateTime', 'Junction', 'Vehicles', 'ID'])
#     df = df.drop_duplicates()
#     df['DateTime'] = pd.to_datetime(df['DateTime'])
#     df['Vehicles'] = pd.to_numeric(df['Vehicles'], errors='coerce')  # Ensure 'Vehicles' column is numeric
#     df = df.dropna(subset=['Vehicles'])
#     df['Junction'] = df['Junction'].astype(int)
#     df['ID'] = df['ID'].apply(lambda x: '{:.0f}'.format(x))
#     df = df.drop_duplicates()
#     df = df.sort_values(by=['Junction', 'DateTime'])
#     df.reset_index(drop=True, inplace=True)
#     df.set_index('DateTime', inplace=True)
#
#     # Handling missing values
#     df.fillna(method='ffill', inplace=True)
#
#     # Extracting date-time features
#     df['Year'] = df.index.year
#     df['Month'] = df.index.month
#     df['Day'] = df.index.day
#     df['Hour'] = df.index.hour
#     df['DayOfWeek'] = df.index.dayofweek
#
#     # Adding cyclical features
#     df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
#     df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
#     df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
#     df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
#
#     # Adding holiday feature
#     cal = calendar()
#     holidays = cal.holidays(start=df.index.min(), end=df.index.max())
#     df['Holiday'] = df.index.isin(holidays).astype(int)
#
#     # Feature scaling
#     scaler = StandardScaler()
#     df['Vehicles_scaled'] = scaler.fit_transform(df[['Vehicles']])
#
#     return df
#
# # Initialize session state
# if 'uploaded_photos' not in st.session_state:
#     st.session_state.uploaded_photos = []
#
# # Main function to run the Streamlit app
# def main():
#     # Set page title and icon
#     st.set_page_config(page_title="Traffic Data Analysis", page_icon=":car:")
#
#     # Load data
#     df = load_data()
#
#     # Create header with title and logos
#     st.title("Traffic Data Analysis")
#
#     # Display logos on header using columns
#     col1, col2 = st.columns([1, 3])
#     with col1:
#         st.image("traffic-police.jpg", width=100, caption="Left Logo")
#
#     with col2:
#         st.image("logo.jpg", width=100, caption="Right Logo")
#
#     # Create sidebar navigation
#     st.sidebar.title("Navigation")
#     selected_tab = st.sidebar.radio("Go to", ["Traffic Data", "Upload Photos", "Settings"])
#
#     # Display selected tab content
#     if selected_tab == "Traffic Data":
#         display_traffic_data(df)
#     elif selected_tab == "Upload Photos":
#         display_upload_photos()
#     elif selected_tab == "Settings":
#         display_settings(df)
#
# # Function to display Traffic Data tab
# def display_traffic_data(df):
#     st.title("Traffic Data Analysis")
#
#     # Check if 'Year' column exists in the DataFrame
#     if 'Year' not in df.columns:
#         st.error("Year column not found in the DataFrame!")
#         return
#
#     # Initialize junctions variable
#     junctions = []
#
#     # Sidebar filters
#     st.sidebar.title("Traffic Data Filters")
#
#     # Checkbox for selecting specific date, year, and hour
#     show_date_checkbox = st.sidebar.checkbox("Filter by Date")
#
#     if show_date_checkbox:
#         st.sidebar.title("Date Filters")
#         year = st.sidebar.selectbox("Select Year", np.append(["Overall"], df['Year'].unique()))
#         month = st.sidebar.selectbox("Select Month", np.append(["Overall"], df['Month'].unique()))
#         day = st.sidebar.selectbox("Select Day", np.append(["Overall"], df['Day'].unique()))
#
#         if year == "Overall":
#             filtered_df = df
#         else:
#             filtered_df = df[df['Year'] == int(year)]
#
#         if month != "Overall":
#             filtered_df = filtered_df[filtered_df['Month'] == int(month)]
#
#         if day != "Overall":
#             filtered_df = filtered_df[filtered_df['Day'] == int(day)]
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}")
#
#     else:
#         st.sidebar.title("Date and Time Filters")
#         year = st.sidebar.selectbox("Select Year", np.append(["Overall"], df['Year'].unique()))
#         month = st.sidebar.selectbox("Select Month", np.append(["Overall"], df['Month'].unique()))
#         day = st.sidebar.selectbox("Select Day", np.append(["Overall"], df['Day'].unique()))
#         hour = st.sidebar.selectbox("Select Hour", np.append(["Overall"], df['Hour'].unique()))
#         junctions = st.sidebar.multiselect("Select Junction", df['Junction'].unique(), default=df['Junction'].unique())
#
#         filtered_df = df
#
#         if year != "Overall":
#             filtered_df = filtered_df[filtered_df['Year'] == int(year)]
#
#         if month != "Overall":
#             filtered_df = filtered_df[filtered_df['Month'] == int(month)]
#
#         if day != "Overall":
#             filtered_df = filtered_df[filtered_df['Day'] == int(day)]
#
#         if hour != "Overall":
#             filtered_df = filtered_df[filtered_df['Hour'] == int(hour)]
#
#         if junctions:
#             filtered_df = filtered_df[filtered_df['Junction'].isin(junctions)]
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}, Hour: {hour}, Junctions: {junctions}")
#
#     # Display filtered data
#     st.write(filtered_df)
#
#     # Visualization of filtered data
#     st.title("Historical Traffic Data - Line Chart")
#
#     if not filtered_df.empty:
#         # Group by Hour and Junction to get the average number of vehicles
#         historical_df = filtered_df.groupby(['Hour', 'Junction'], as_index=False)['Vehicles'].mean()
#
#         # Create line chart using Plotly Express
#         fig = px.line(historical_df, x='Hour', y='Vehicles', color='Junction',
#                       title='Average Hourly Traffic Data',
#                       labels={'Hour': 'Hour of Day', 'Vehicles': 'Average Number of Vehicles'})
#         st.plotly_chart(fig)
#
# # Function to display Upload Photos tab
# def display_upload_photos():
#     st.title("Upload Photos")
#
#     # Allow users to upload photos
#     uploaded_files = st.file_uploader("Choose a photo", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
#
#     # Display uploaded photos
#     if uploaded_files:
#         for uploaded_file in uploaded_files:
#             # Resize the image to a smaller size for display
#             img = Image.open(uploaded_file)
#             img = img.resize((200, 200))  # Adjust the size as needed
#             st.image(img, caption=uploaded_file.name)
#
#         # Append uploaded files to session state
#         st.session_state.uploaded_photos.extend(uploaded_files)
#
#         # Display success message
#         st.success("Photos uploaded successfully!")
#
#     # Display option to download uploaded photos
#     if st.session_state.uploaded_photos:
#         st.title("Download Uploaded Photos")
#
#         # Display links for downloading uploaded photos
#         for i, uploaded_file in enumerate(st.session_state.uploaded_photos, start=1):
#             photo_name = uploaded_file.name  # Get the filename directly
#             st.markdown(f"**{photo_name}**: ")
#             st.markdown(get_image_download_link(uploaded_file, photo_name), unsafe_allow_html=True)
#
# # Function to get download link for an image
# def get_image_download_link(img, text):
#     buffered = BytesIO()
#     img.save(buffered, format="JPEG")
#     img_str = base64.b64encode(buffered.getvalue()).decode()
#     href = f'<a href="data:file/jpg;base64,{img_str}" download="{text}.jpg">Download {text}</a>'
#     return href
#
# # Function to display Settings tab
# def display_settings(df):
#     st.title("Settings")
#
#     st.header("Visualization Settings")
#     # Example: Adjusting line chart granularity
#     line_chart_granularity = st.selectbox("Line Chart Granularity", ["Hour", "Day", "Month"])
#     st.write(f"Selected Line Chart Granularity: {line_chart_granularity}")
#
#     # Example: Adjusting color scheme
#     color_scheme = st.selectbox("Color Scheme", ["Plotly Express", "Streamlit Default", "Custom"])
#     st.write(f"Selected Color Scheme: {color_scheme}")
#
#     # Example: Adjusting plot size
#     plot_size = st.slider("Plot Size", min_value=200, max_value=1000, value=500, step=50)
#     st.write(f"Selected Plot Size: {plot_size}")
#
#     st.header("Data Filtering Defaults")
#     # Example: Default year selection
#     default_year = st.selectbox("Default Year", df['Year'].unique())
#     st.write(f"Selected Default Year: {default_year}")
#
# # Run the main function
# if __name__ == "__main__":
#     main()

# import streamlit as st
# import pandas as pd
# import numpy as np
# from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
# from sklearn.preprocessing import StandardScaler
# import plotly.express as px
# from PIL import Image
# import base64
# from io import BytesIO  # Import BytesIO for handling bytes data
#
# # Load and preprocess data
# @st.cache_data
# def load_data():
#     # Load the data (replace with your data loading logic)
#     df = pd.read_csv("uncleaned_traffic.csv")
#
#     # Your data preprocessing steps go here
#     # Data cleaning
#     df = df.dropna(subset=['DateTime', 'Junction', 'Vehicles', 'ID'])
#     df = df.drop_duplicates()
#     df['DateTime'] = pd.to_datetime(df['DateTime'])
#     df['Vehicles'] = pd.to_numeric(df['Vehicles'], errors='coerce')  # Ensure 'Vehicles' column is numeric
#     df = df.dropna(subset=['Vehicles'])
#     df['Junction'] = df['Junction'].astype(int)
#     df['ID'] = df['ID'].apply(lambda x: '{:.0f}'.format(x))
#     df = df.drop_duplicates()
#     df = df.sort_values(by=['Junction', 'DateTime'])
#     df.reset_index(drop=True, inplace=True)
#     df.set_index('DateTime', inplace=True)
#
#     # Handling missing values
#     df.fillna(method='ffill', inplace=True)
#
#     # Extracting date-time features
#     df['Year'] = df.index.year
#     df['Month'] = df.index.month
#     df['Day'] = df.index.day
#     df['Hour'] = df.index.hour
#     df['DayOfWeek'] = df.index.dayofweek
#
#     # Adding cyclical features
#     df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
#     df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
#     df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
#     df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
#
#     # Adding holiday feature
#     cal = calendar()
#     holidays = cal.holidays(start=df.index.min(), end=df.index.max())
#     df['Holiday'] = df.index.isin(holidays).astype(int)
#
#     # Feature scaling
#     scaler = StandardScaler()
#     df['Vehicles_scaled'] = scaler.fit_transform(df[['Vehicles']])
#
#     return df
#
# # Initialize session state
# if 'uploaded_photos' not in st.session_state:
#     st.session_state.uploaded_photos = []
#
# # Main function to run the Streamlit app
# def main():
#     # Set page title and icon
#     st.set_page_config(page_title="Traffic Data Analysis", page_icon=":car:")
#
#     # Load data
#     df = load_data()
#
#     # Create header with title and logos
#     st.title("Traffic Data Analysis")
#
#     # Display logos on header using columns
#     col1, col2 = st.columns([4, 1])  # Adjust column ratios as needed
#     with col1:
#         st.title("Traffic Data Analysis")  # Big title
#     with col2:
#         st.image("logo.jpg", width=100)  # Logo image
#
#     # Create sidebar navigation
#     st.sidebar.title("Navigation")
#     selected_tab = st.sidebar.radio("Go to", ["Traffic Data", "Upload Photos", "Settings"])
#
#     # Display selected tab content
#     if selected_tab == "Traffic Data":
#         display_traffic_data(df)
#     elif selected_tab == "Upload Photos":
#         display_upload_photos()
#     elif selected_tab == "Settings":
#         display_settings(df)
#
# # Function to display Traffic Data tab
# def display_traffic_data(df):
#     st.title("Traffic Data Analysis")
#
#     # Check if 'Year' column exists in the DataFrame
#     if 'Year' not in df.columns:
#         st.error("Year column not found in the DataFrame!")
#         return
#
#     # Initialize junctions variable
#     junctions = []
#
#     # Sidebar filters
#     st.sidebar.title("Traffic Data Filters")
#
#     # Checkbox for selecting specific date, year, and hour
#     show_date_checkbox = st.sidebar.checkbox("Filter by Date")
#
#     if show_date_checkbox:
#         st.sidebar.title("Date Filters")
#         year = st.sidebar.selectbox("Select Year", np.append(["Overall"], df['Year'].unique()))
#         month = st.sidebar.selectbox("Select Month", np.append(["Overall"], df['Month'].unique()))
#         day = st.sidebar.selectbox("Select Day", np.append(["Overall"], df['Day'].unique()))
#
#         if year == "Overall":
#             filtered_df = df
#         else:
#             filtered_df = df[df['Year'] == int(year)]
#
#         if month != "Overall":
#             filtered_df = filtered_df[filtered_df['Month'] == int(month)]
#
#         if day != "Overall":
#             filtered_df = filtered_df[filtered_df['Day'] == int(day)]
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}")
#
#     else:
#         st.sidebar.title("Date and Time Filters")
#         year = st.sidebar.selectbox("Select Year", np.append(["Overall"], df['Year'].unique()))
#         month = st.sidebar.selectbox("Select Month", np.append(["Overall"], df['Month'].unique()))
#         day = st.sidebar.selectbox("Select Day", np.append(["Overall"], df['Day'].unique()))
#         hour = st.sidebar.selectbox("Select Hour", np.append(["Overall"], df['Hour'].unique()))
#         junctions = st.sidebar.multiselect("Select Junction", df['Junction'].unique(), default=df['Junction'].unique())
#
#         filtered_df = df
#
#         if year != "Overall":
#             filtered_df = filtered_df[filtered_df['Year'] == int(year)]
#
#         if month != "Overall":
#             filtered_df = filtered_df[filtered_df['Month'] == int(month)]
#
#         if day != "Overall":
#             filtered_df = filtered_df[filtered_df['Day'] == int(day)]
#
#         if hour != "Overall":
#             filtered_df = filtered_df[filtered_df['Hour'] == int(hour)]
#
#         if junctions:
#             filtered_df = filtered_df[filtered_df['Junction'].isin(junctions)]
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}, Hour: {hour}, Junctions: {junctions}")
#
#     # Display filtered data
#     st.write(filtered_df)
#
#     # Visualization of filtered data
#     st.title("Historical Traffic Data - Line Chart")
#
#     if not filtered_df.empty:
#         # Group by Hour and Junction to get the average number of vehicles
#         historical_df = filtered_df.groupby(['Hour', 'Junction'], as_index=False)['Vehicles'].mean()
#
#         # Create line chart using Plotly Express
#         fig = px.line(historical_df, x='Hour', y='Vehicles', color='Junction',
#                       title='Average Hourly Traffic Data',
#                       labels={'Hour': 'Hour of Day', 'Vehicles': 'Average Number of Vehicles'})
#         st.plotly_chart(fig)
#
# # Function to display Upload Photos tab
# def display_upload_photos():
#     st.title("Upload Photos")
#
#     # Allow users to upload photos
#     uploaded_files = st.file_uploader("Choose a photo", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
#
#     # Display uploaded photos
#     if uploaded_files:
#         for uploaded_file in uploaded_files:
#             # Resize the image to a smaller size for display
#             img = Image.open(uploaded_file)
#             img = img.resize((200, 200))  # Adjust the size as needed
#             st.image(img, caption=uploaded_file.name)
#
#         # Append uploaded files to session state
#         st.session_state.uploaded_photos.extend(uploaded_files)
#
#         # Display success message
#         st.success("Photos uploaded successfully!")
#
#     # Display option to download uploaded photos
#     if st.session_state.uploaded_photos:
#         st.title("Download Uploaded Photos")
#
#         # Display links for downloading uploaded photos
#         for i, uploaded_file in enumerate(st.session_state.uploaded_photos, start=1):
#             photo_name = uploaded_file.name  # Get the filename directly
#             st.markdown(f"**{photo_name}**: ")
#             st.markdown(get_image_download_link(uploaded_file, photo_name), unsafe_allow_html=True)
#
# # Function to get download link for an image
# def get_image_download_link(img, text):
#     img_read = img.read()  # Read the file here
#     buffered = BytesIO(img_read)
#     img = Image.open(buffered)
#     img.save(buffered, format="JPEG")
#     img_str = base64.b64encode(buffered.getvalue()).decode()
#     href = f'<a href="data:file/jpg;base64,{img_str}" download="{text}.jpg">Download {text}</a>'
#     return href
#
# # Function to display Settings tab
# def display_settings(df):
#     st.title("Settings")
#
#     st.header("Visualization Settings")
#     # Example: Adjusting line chart granularity
#     line_chart_granularity = st.selectbox("Line Chart Granularity", ["Hour", "Day", "Month"])
#     st.write(f"Selected Line Chart Granularity: {line_chart_granularity}")
#
#     # Example: Adjusting color scheme
#     color_scheme = st.selectbox("Color Scheme", ["Plotly Express", "Streamlit Default", "Custom"])
#     st.write(f"Selected Color Scheme: {color_scheme}")
#
#     # Example: Adjusting plot size
#     plot_size = st.slider("Plot Size", min_value=200, max_value=1000, value=500, step=50)
#     st.write(f"Selected Plot Size: {plot_size}")
#
#     st.header("Data Filtering Defaults")
#     # Example: Default year selection
#     default_year = st.selectbox("Default Year", df['Year'].unique())
#     st.write(f"Selected Default Year: {default_year}")
#
# # Run the main function
# if __name__ == "__main__":
#     main()

# import streamlit as st
# import pandas as pd
# import numpy as np
# from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
# from sklearn.preprocessing import StandardScaler
# import plotly.express as px
# from PIL import Image
# import base64
# from io import BytesIO  # Import BytesIO for handling bytes data
#
# # Load and preprocess data
# @st.cache
# def load_data():
#     # Load the data (replace with your data loading logic)
#     df = pd.read_csv("uncleaned_traffic.csv")
#
#     # Your data preprocessing steps go here
#     # Data cleaning
#     df = df.dropna(subset=['DateTime', 'Junction', 'Vehicles', 'ID'])
#     df = df.drop_duplicates()
#     df['DateTime'] = pd.to_datetime(df['DateTime'])
#     df['Vehicles'] = pd.to_numeric(df['Vehicles'], errors='coerce')  # Ensure 'Vehicles' column is numeric
#     df = df.dropna(subset=['Vehicles'])
#     df['Junction'] = df['Junction'].astype(int)
#     df['ID'] = df['ID'].apply(lambda x: '{:.0f}'.format(x))
#     df = df.drop_duplicates()
#     df = df.sort_values(by=['Junction', 'DateTime'])
#     df.reset_index(drop=True, inplace=True)
#     df.set_index('DateTime', inplace=True)
#
#     # Handling missing values
#     df.fillna(method='ffill', inplace=True)
#
#     # Extracting date-time features
#     df['Year'] = df.index.year
#     df['Month'] = df.index.month
#     df['Day'] = df.index.day
#     df['Hour'] = df.index.hour
#     df['DayOfWeek'] = df.index.dayofweek
#
#     # Adding cyclical features
#     df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
#     df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
#     df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
#     df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
#
#     # Adding holiday feature
#     cal = calendar()
#     holidays = cal.holidays(start=df.index.min(), end=df.index.max())
#     df['Holiday'] = df.index.isin(holidays).astype(int)
#
#     # Feature scaling
#     scaler = StandardScaler()
#     df['Vehicles_scaled'] = scaler.fit_transform(df[['Vehicles']])
#
#     return df
#
#
# # Initialize session state
# if 'uploaded_photos' not in st.session_state:
#     st.session_state.uploaded_photos = []
#
#
# # Main function to run the Streamlit app
# def main():
#     # Set page title and icon
#     st.set_page_config(page_title="Traffic Data Analysis", page_icon=":car:")
#
#     # Load data
#     df = load_data()
#
#     # Create header with title and logos
#     st.title("Traffic Data Analysis")
#
#     # Display logos on header using columns
#     col1, col2 = st.columns([1, 3])
#     with col1:
#         st.image("traffic-police.jpg", width=100, caption="Left Logo")
#
#     with col2:
#         st.image("logo.jpg", width=100, caption="Right Logo")
#
#     # Create sidebar navigation
#     st.sidebar.title("Navigation")
#     selected_tab = st.sidebar.radio("Go to", ["Traffic Data", "Upload Photos", "Settings"])
#
#     # Display selected tab content
#     if selected_tab == "Traffic Data":
#         display_traffic_data(df)
#     elif selected_tab == "Upload Photos":
#         display_upload_photos()
#     elif selected_tab == "Settings":
#         display_settings(df)
#
#
# # Function to display Traffic Data tab
# def display_traffic_data(df):
#     st.title("Traffic Data Analysis")
#
#     # Check if 'Year' column exists in the DataFrame
#     if 'Year' not in df.columns:
#         st.error("Year column not found in the DataFrame!")
#         return
#
#     # Initialize junctions variable
#     junctions = []
#
#     # Sidebar filters
#     st.sidebar.title("Traffic Data Filters")
#
#     # Checkbox for selecting specific date, year, and hour
#     show_date_checkbox = st.sidebar.checkbox("Filter by Date")
#
#     if show_date_checkbox:
#         st.sidebar.title("Date Filters")
#         year = st.sidebar.selectbox("Select Year", np.append(["Overall"], df['Year'].unique()))
#         month = st.sidebar.selectbox("Select Month", np.append(["Overall"], df['Month'].unique()))
#         day = st.sidebar.selectbox("Select Day", np.append(["Overall"], df['Day'].unique()))
#
#         if year == "Overall":
#             filtered_df = df
#         else:
#             filtered_df = df[df['Year'] == int(year)]
#
#         if month != "Overall":
#             filtered_df = filtered_df[filtered_df['Month'] == int(month)]
#
#         if day != "Overall":
#             filtered_df = filtered_df[filtered_df['Day'] == int(day)]
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}")
#
#     else:
#         st.sidebar.title("Date and Time Filters")
#         year = st.sidebar.selectbox("Select Year", np.append(["Overall"], df['Year'].unique()))
#         month = st.sidebar.selectbox("Select Month", np.append(["Overall"], df['Month'].unique()))
#         day = st.sidebar.selectbox("Select Day", np.append(["Overall"], df['Day'].unique()))
#         hour = st.sidebar.selectbox("Select Hour", np.append(["Overall"], df['Hour'].unique()))
#         junctions = st.sidebar.multiselect("Select Junction", df['Junction'].unique(), default=df['Junction'].unique())
#
#         filtered_df = df
#
#         if year != "Overall":
#             filtered_df = filtered_df[filtered_df['Year'] == int(year)]
#
#         if month != "Overall":
#             filtered_df = filtered_df[filtered_df['Month'] == int(month)]
#
#         if day != "Overall":
#             filtered_df = filtered_df[filtered_df['Day'] == int(day)]
#
#         if hour != "Overall":
#             filtered_df = filtered_df[filtered_df['Hour'] == int(hour)]
#
#         if junctions:
#             filtered_df = filtered_df[filtered_df['Junction'].isin(junctions)]
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}, Hour: {hour}, Junctions: {junctions}")
#
#     # Display filtered data
#     st.write(filtered_df)
#
#     # Visualization of filtered data
#     st.title("Historical Traffic Data - Line Chart")
#
#     if not filtered_df.empty:
#         # Group by Hour and Junction to get the average number of vehicles
#         historical_df = filtered_df.groupby(['Hour', 'Junction'], as_index=False)['Vehicles'].mean()
#
#         # Create line chart using Plotly Express
#         fig = px.line(historical_df, x='Hour', y='Vehicles', color='Junction',
#                       title='Average Hourly Traffic Data',
#                       labels={'Hour': 'Hour of Day', 'Vehicles': 'Average Number of Vehicles'})
#         st.plotly_chart(fig)
#
#
# # Function to display Upload Photos tab
# def display_upload_photos():
#     st.title("Upload Photos")
#
#     # Allow users to upload photos
#     uploaded_files = st.file_uploader("Choose a photo", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
#
#     # Display uploaded photos
#     if uploaded_files:
#         for uploaded_file in uploaded_files:
#             # Resize the image to a smaller size for display
#             img = Image.open(uploaded_file)
#             img = img.resize((200, 200))  # Adjust the size as needed
#             st.image(img, caption=uploaded_file.name)
#
#             # Append uploaded files to session state
#             st.session_state.uploaded_photos.append(uploaded_file)
#
#         # Display success message
#         st.success("Photos uploaded successfully!")
#
#     # Display option to download uploaded photos
#     if st.session_state.uploaded_photos:
#         st.title("Download Uploaded Photos")
#
#         # Display links for downloading uploaded photos
#         for i, uploaded_file in enumerate(st.session_state.uploaded_photos, start=1):
#             photo_name = uploaded_file.name
#             st.markdown(f"**{photo_name}**: ")
#             st.markdown(get_image_download_link(uploaded_file, photo_name), unsafe_allow_html=True)
#
#
# # Function to get download link for an image
# def get_image_download_link(img, text):
#     img_read = img.read()  # Read the file data
#     img_buffer = BytesIO(img_read)  # Create a BytesIO buffer
#     img = Image.open(img_buffer)  # Open the image with PIL
#
#     # Resize the image to a smaller size (optional)
#     img = img.resize((200, 200))
#
#     # Save the image to a BytesIO buffer as JPEG
#     img_buffer = BytesIO()
#     img.save(img_buffer, format="JPEG")
#     img_str = base64.b64encode(img_buffer.getvalue()).decode()
#
#     # Generate download link
#     href = f'<a href="data:file/jpg;base64,{img_str}" download="{text}.jpg">Download {text}</a>'
#     return href
#
#
# # Function to display Settings tab
# def display_settings(df):
#     st.title("Settings")
#
#     st.header("Visualization Settings")
#     # Example: Adjusting line chart granularity
#     line_chart_granularity = st.selectbox("Line Chart Granularity", ["Hour", "Day", "Month"])
#     st.write(f"Selected Line Chart Granularity: {line_chart_granularity}")
#
#     # Example: Adjusting color scheme
#     color_scheme = st.selectbox("Color Scheme", ["Plotly Express", "Streamlit Default", "Custom"])
#     st.write(f"Selected Color Scheme: {color_scheme}")
#
#     # Example: Adjusting plot size
#     plot_size = st.slider("Plot Size", min_value=200, max_value=1000, value=500, step=50)
#     st.write(f"Selected Plot Size: {plot_size}")
#
#     st.header("Data Filtering Defaults")
#     # Example: Default year selection
#     default_year = st.selectbox("Default Year", df['Year'].unique())
#     st.write(f"Selected Default Year: {default_year}")
#
#
# # Run the main function
# if __name__ == "__main__":
#     main()
#
# import streamlit as st
# import pandas as pd
# import numpy as np
# from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
# from sklearn.preprocessing import StandardScaler
# import plotly.express as px
# from PIL import Image
# import base64
# from io import BytesIO  # Import BytesIO for handling bytes data
#
# # Load and preprocess data
# @st.cache
# def load_data():
#     # Load the data (replace with your data loading logic)
#     df = pd.read_csv("uncleaned_traffic.csv")
#
#     # Your data preprocessing steps go here
#     # Data cleaning
#     df = df.dropna(subset=['DateTime', 'Junction', 'Vehicles', 'ID'])
#     df = df.drop_duplicates()
#     df['DateTime'] = pd.to_datetime(df['DateTime'])
#     df['Vehicles'] = pd.to_numeric(df['Vehicles'], errors='coerce')  # Ensure 'Vehicles' column is numeric
#     df = df.dropna(subset=['Vehicles'])
#     df['Junction'] = df['Junction'].astype(int)
#     df['ID'] = df['ID'].apply(lambda x: '{:.0f}'.format(x))
#     df = df.drop_duplicates()
#     df = df.sort_values(by=['Junction', 'DateTime'])
#     df.reset_index(drop=True, inplace=True)
#     df.set_index('DateTime', inplace=True)
#
#     # Handling missing values
#     df.fillna(method='ffill', inplace=True)
#
#     # Extracting date-time features
#     df['Year'] = df.index.year
#     df['Month'] = df.index.month
#     df['Day'] = df.index.day
#     df['Hour'] = df.index.hour
#     df['DayOfWeek'] = df.index.dayofweek
#
#     # Adding cyclical features
#     df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
#     df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
#     df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
#     df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
#
#     # Adding holiday feature
#     cal = calendar()
#     holidays = cal.holidays(start=df.index.min(), end=df.index.max())
#     df['Holiday'] = df.index.isin(holidays).astype(int)
#
#     # Feature scaling
#     scaler = StandardScaler()
#     df['Vehicles_scaled'] = scaler.fit_transform(df[['Vehicles']])
#
#     return df
#
#
# # Initialize session state
# if 'uploaded_photos' not in st.session_state:
#     st.session_state.uploaded_photos = []
#
#
# # Main function to run the Streamlit app
# def main():
#     # Set page title and icon
#     st.set_page_config(page_title="Traffic Data Analysis", page_icon=":car:")
#
#     # Load data
#     df = load_data()
#
#     # Create header with title and logos
#     st.title("Traffic Data Analysis")
#
#     # Display logos on header using columns
#     col1, col2 = st.columns([1, 3])
#     with col1:
#         st.image("traffic-police.jpg", width=100, caption="Left Logo")
#
#     with col2:
#         st.image("logo.jpg", width=100, caption="Right Logo")
#
#     # Create sidebar navigation
#     st.sidebar.title("Navigation")
#     selected_tab = st.sidebar.radio("Go to", ["Traffic Data", "Upload Photos", "Settings"])
#
#     # Display selected tab content
#     if selected_tab == "Traffic Data":
#         display_traffic_data(df)
#     elif selected_tab == "Upload Photos":
#         display_upload_photos()
#     elif selected_tab == "Settings":
#         display_settings(df)
#
#
# # Function to display Traffic Data tab
# def display_traffic_data(df):
#     st.title("Traffic Data Analysis")
#
#     # Check if 'Year' column exists in the DataFrame
#     if 'Year' not in df.columns:
#         st.error("Year column not found in the DataFrame!")
#         return
#
#     # Initialize junctions variable
#     junctions = []
#
#     # Sidebar filters
#     st.sidebar.title("Traffic Data Filters")
#
#     # Checkbox for selecting specific date, year, and hour
#     show_date_checkbox = st.sidebar.checkbox("Filter by Date")
#
#     if show_date_checkbox:
#         st.sidebar.title("Date Filters")
#         year = st.sidebar.selectbox("Select Year", np.append(["Overall"], df['Year'].unique()))
#         month = st.sidebar.selectbox("Select Month", np.append(["Overall"], df['Month'].unique()))
#         day = st.sidebar.selectbox("Select Day", np.append(["Overall"], df['Day'].unique()))
#
#         if year == "Overall":
#             filtered_df = df
#         else:
#             filtered_df = df[df['Year'] == int(year)]
#
#         if month != "Overall":
#             filtered_df = filtered_df[filtered_df['Month'] == int(month)]
#
#         if day != "Overall":
#             filtered_df = filtered_df[filtered_df['Day'] == int(day)]
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}")
#
#     else:
#         st.sidebar.title("Date and Time Filters")
#         year = st.sidebar.selectbox("Select Year", np.append(["Overall"], df['Year'].unique()))
#         month = st.sidebar.selectbox("Select Month", np.append(["Overall"], df['Month'].unique()))
#         day = st.sidebar.selectbox("Select Day", np.append(["Overall"], df['Day'].unique()))
#         hour = st.sidebar.selectbox("Select Hour", np.append(["Overall"], df['Hour'].unique()))
#         junctions = st.sidebar.multiselect("Select Junction", df['Junction'].unique(), default=df['Junction'].unique())
#
#         filtered_df = df
#
#         if year != "Overall":
#             filtered_df = filtered_df[filtered_df['Year'] == int(year)]
#
#         if month != "Overall":
#             filtered_df = filtered_df[filtered_df['Month'] == int(month)]
#
#         if day != "Overall":
#             filtered_df = filtered_df[filtered_df['Day'] == int(day)]
#
#         if hour != "Overall":
#             filtered_df = filtered_df[filtered_df['Hour'] == int(hour)]
#
#         if junctions:
#             filtered_df = filtered_df[filtered_df['Junction'].isin(junctions)]
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}, Hour: {hour}, Junctions: {junctions}")
#
#     # Display filtered data
#     st.write(filtered_df)
#
#     # Visualization of filtered data
#     st.title("Historical Traffic Data - Line Chart")
#
#     if not filtered_df.empty:
#         # Group by Hour and Junction to get the average number of vehicles
#         historical_df = filtered_df.groupby(['Hour', 'Junction'], as_index=False)['Vehicles'].mean()
#
#         # Create line chart using Plotly Express
#         fig = px.line(historical_df, x='Hour', y='Vehicles', color='Junction',
#                       title='Average Hourly Traffic Data',
#                       labels={'Hour': 'Hour of Day', 'Vehicles': 'Average Number of Vehicles'})
#         st.plotly_chart(fig)
#
#
# # Function to display Upload Photos tab
# def display_upload_photos():
#     st.title("Upload Photos")
#
#     # Allow users to upload photos
#     uploaded_files = st.file_uploader("Choose a photo", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
#
#     # Display uploaded photos
#     if uploaded_files:
#         for uploaded_file in uploaded_files:
#             # Resize the image to a smaller size for display
#             img = Image.open(uploaded_file)
#             img = img.resize((200, 200))  # Adjust the size as needed
#             st.image(img, caption=uploaded_file.name)
#
#             # Append uploaded files to session state
#             st.session_state.uploaded_photos.append(uploaded_file)
#
#         # Display success message
#         st.success("Photos uploaded successfully!")
#
#     # Display option to download uploaded photos
#     if st.session_state.uploaded_photos:
#         st.title("Download Uploaded Photos")
#
#         # Display links for downloading uploaded photos
#         for i, uploaded_file in enumerate(st.session_state.uploaded_photos, start=1):
#             photo_name = uploaded_file.name
#             st.markdown(f"**{photo_name}**: ")
#             st.markdown(get_image_download_link(uploaded_file, photo_name), unsafe_allow_html=True)
#
#
# # Function to get download link for an image
# def get_image_download_link(img, text):
#     img_read = img.read()  # Read the file data
#     img = Image.open(BytesIO(img_read))  # Open the image with PIL
#
#     # Resize the image to a smaller size (optional)
#     img = img.resize((200, 200))
#
#     # Save the image to a BytesIO buffer as JPEG
#     img_buffer = BytesIO()
#     img.save(img_buffer, format="JPEG")
#     img_str = base64.b64encode(img_buffer.getvalue()).decode()
#
#     # Generate download link
#     href = f'<a href="data:file/jpg;base64,{img_str}" download="{text}.jpg">Download {text}</a>'
#     return href
#
#
# # Function to display Settings tab
# def display_settings(df):
#     st.title("Settings")
#
#     st.header("Visualization Settings")
#     # Example: Adjusting line chart granularity
#     line_chart_granularity = st.selectbox("Line Chart Granularity", ["Hour", "Day", "Month"])
#     st.write(f"Selected Line Chart Granularity: {line_chart_granularity}")
#
#     # Example: Adjusting color scheme
#     color_scheme = st.selectbox("Color Scheme", ["Plotly Express", "Streamlit Default", "Custom"])
#     st.write(f"Selected Color Scheme: {color_scheme}")
#
#     # Example: Adjusting plot size
#     plot_size = st.slider("Plot Size", min_value=200, max_value=1000, value=500, step=50)
#     st.write(f"Selected Plot Size: {plot_size}")
#
#     st.header("Data Filtering Defaults")
#     # Example: Default year selection
#     default_year = st.selectbox("Default Year", df['Year'].unique())
#     st.write(f"Selected Default Year: {default_year}")
#
#
# # Run the main function
# if __name__ == "__main__":
#     main()

# import streamlit as st
# import pandas as pd
# import numpy as np
# from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
# from sklearn.preprocessing import StandardScaler
# import plotly.express as px
# from PIL import Image
# import base64
# from io import BytesIO  # Import BytesIO for handling bytes data
#
# # Load and preprocess data
# @st.cache
# def load_data():
#     # Load the data (replace with your data loading logic)
#     df = pd.read_csv("uncleaned_traffic.csv")
#
#     # Your data preprocessing steps go here
#     # Data cleaning
#     df = df.dropna(subset=['DateTime', 'Junction', 'Vehicles', 'ID'])
#     df = df.drop_duplicates()
#     df['DateTime'] = pd.to_datetime(df['DateTime'])
#     df['Vehicles'] = pd.to_numeric(df['Vehicles'], errors='coerce')  # Ensure 'Vehicles' column is numeric
#     df = df.dropna(subset=['Vehicles'])
#     df['Junction'] = df['Junction'].astype(int)
#     df['ID'] = df['ID'].apply(lambda x: '{:.0f}'.format(x))
#     df = df.drop_duplicates()
#     df = df.sort_values(by=['Junction', 'DateTime'])
#     df.reset_index(drop=True, inplace=True)
#     df.set_index('DateTime', inplace=True)
#
#     # Handling missing values
#     df.fillna(method='ffill', inplace=True)
#
#     # Extracting date-time features
#     df['Year'] = df.index.year
#     df['Month'] = df.index.month
#     df['Day'] = df.index.day
#     df['Hour'] = df.index.hour
#     df['DayOfWeek'] = df.index.dayofweek
#
#     # Adding cyclical features
#     df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
#     df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
#     df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
#     df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
#
#     # Adding holiday feature
#     cal = calendar()
#     holidays = cal.holidays(start=df.index.min(), end=df.index.max())
#     df['Holiday'] = df.index.isin(holidays).astype(int)
#
#     # Feature scaling
#     scaler = StandardScaler()
#     df['Vehicles_scaled'] = scaler.fit_transform(df[['Vehicles']])
#
#     return df
#
#
# # Initialize session state
# if 'uploaded_photos' not in st.session_state:
#     st.session_state.uploaded_photos = []
#
#
# # Main function to run the Streamlit app
# def main():
#     # Set page title and icon
#     st.set_page_config(page_title="Traffic Data Analysis", page_icon=":car:")
#
#     # Load data
#     df = load_data()
#
#     # Create header with title and logos
#     st.title("Traffic Data Analysis")
#
#     # Display logos on header using columns
#     col1, col2 = st.columns([1, 3])
#     with col1:
#         st.image("traffic-police.jpg", width=100, caption="Left Logo")
#
#     with col2:
#         st.image("logo.jpg", width=100, caption="Right Logo")
#
#     # Create sidebar navigation
#     st.sidebar.title("Navigation")
#     selected_tab = st.sidebar.radio("Go to", ["Traffic Data", "Upload Photos", "Settings"])
#
#     # Display selected tab content
#     if selected_tab == "Traffic Data":
#         display_traffic_data(df)
#     elif selected_tab == "Upload Photos":
#         display_upload_photos()
#     elif selected_tab == "Settings":
#         display_settings(df)
#
#
# # Function to display Traffic Data tab
# def display_traffic_data(df):
#     st.title("Traffic Data Analysis")
#
#     # Check if 'Year' column exists in the DataFrame
#     if 'Year' not in df.columns:
#         st.error("Year column not found in the DataFrame!")
#         return
#
#     # Initialize junctions variable
#     junctions = []
#
#     # Sidebar filters
#     st.sidebar.title("Traffic Data Filters")
#
#     # Checkbox for selecting specific date, year, and hour
#     show_date_checkbox = st.sidebar.checkbox("Filter by Date")
#
#     if show_date_checkbox:
#         st.sidebar.title("Date Filters")
#         year = st.sidebar.selectbox("Select Year", np.append(["Overall"], df['Year'].unique()))
#         month = st.sidebar.selectbox("Select Month", np.append(["Overall"], df['Month'].unique()))
#         day = st.sidebar.selectbox("Select Day", np.append(["Overall"], df['Day'].unique()))
#
#         if year == "Overall":
#             filtered_df = df
#         else:
#             filtered_df = df[df['Year'] == int(year)]
#
#         if month != "Overall":
#             filtered_df = filtered_df[filtered_df['Month'] == int(month)]
#
#         if day != "Overall":
#             filtered_df = filtered_df[filtered_df['Day'] == int(day)]
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}")
#
#     else:
#         st.sidebar.title("Date and Time Filters")
#         year = st.sidebar.selectbox("Select Year", np.append(["Overall"], df['Year'].unique()))
#         month = st.sidebar.selectbox("Select Month", np.append(["Overall"], df['Month'].unique()))
#         day = st.sidebar.selectbox("Select Day", np.append(["Overall"], df['Day'].unique()))
#         hour = st.sidebar.selectbox("Select Hour", np.append(["Overall"], df['Hour'].unique()))
#         junctions = st.sidebar.multiselect("Select Junction", df['Junction'].unique(), default=df['Junction'].unique())
#
#         filtered_df = df
#
#         if year != "Overall":
#             filtered_df = filtered_df[filtered_df['Year'] == int(year)]
#
#         if month != "Overall":
#             filtered_df = filtered_df[filtered_df['Month'] == int(month)]
#
#         if day != "Overall":
#             filtered_df = filtered_df[filtered_df['Day'] == int(day)]
#
#         if hour != "Overall":
#             filtered_df = filtered_df[filtered_df['Hour'] == int(hour)]
#
#         if junctions:
#             filtered_df = filtered_df[filtered_df['Junction'].isin(junctions)]
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}, Hour: {hour}, Junctions: {junctions}")
#
#     # Display filtered data
#     st.write(filtered_df)
#
#     # Visualization of filtered data
#     st.title("Historical Traffic Data - Line Chart")
#
#     if not filtered_df.empty:
#         # Group by Hour and Junction to get the average number of vehicles
#         historical_df = filtered_df.groupby(['Hour', 'Junction'], as_index=False)['Vehicles'].mean()
#
#         # Create line chart using Plotly Express
#         fig = px.line(historical_df, x='Hour', y='Vehicles', color='Junction',
#                       title='Average Hourly Traffic Data',
#                       labels={'Hour': 'Hour of Day', 'Vehicles': 'Average Number of Vehicles'})
#         st.plotly_chart(fig)
#
#
# # Function to display Upload Photos tab
# def display_upload_photos():
#     st.title("Upload Photos")
#
#     # Allow users to upload photos
#     uploaded_files = st.file_uploader("Choose a photo", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
#
#     # Display uploaded photos
#     if uploaded_files:
#         for uploaded_file in uploaded_files:
#             try:
#                 # Open and resize the image
#                 img = Image.open(uploaded_file)
#                 img = img.resize((200, 200))  # Adjust the size as needed
#                 st.image(img, caption=uploaded_file.name)
#
#                 # Append uploaded files to session state
#                 st.session_state.uploaded_photos.append(uploaded_file)
#
#             except UnidentifiedImageError as e:
#                 st.warning(f"Unable to identify image file {uploaded_file.name}: {e}")
#
#         # Display success message
#         st.success("Photos uploaded successfully!")
#
#     # Display option to download uploaded photos
#     if st.session_state.uploaded_photos:
#         st.title("Download Uploaded Photos")
#
#         # Display links for downloading uploaded photos
#         for i, uploaded_file in enumerate(st.session_state.uploaded_photos, start=1):
#             photo_name = uploaded_file.name
#             st.markdown(f"**{photo_name}**: ")
#             st.markdown(get_image_download_link(uploaded_file, photo_name), unsafe_allow_html=True)
#
#
# # Function to get download link for an image
# def get_image_download_link(img, text):
#     img_read = img.read()  # Read the file data
#     img = Image.open(img)  # Open the image with PIL (already opened in display_upload_photos)
#
#     # Resize the image to a smaller size (optional)
#     img = img.resize((200, 200))
#
#     # Save the image to a BytesIO buffer as JPEG
#     img_buffer = BytesIO()
#     img.save(img_buffer, format="JPEG")
#     img_str = base64.b64encode(img_buffer.getvalue()).decode()
#
#     # Generate download link
#     href = f'<a href="data:file/jpg;base64,{img_str}" download="{text}.jpg">Download {text}</a>'
#     return href
#
#
# # Function to display Settings tab
# def display_settings(df):
#     st.title("Settings")
#
#     st.header("Visualization Settings")
#     # Example: Adjusting line chart granularity
#     line_chart_granularity = st.selectbox("Line Chart Granularity", ["Hour", "Day", "Month"])
#     st.write(f"Selected Line Chart Granularity: {line_chart_granularity}")
#
#     # Example: Adjusting color scheme
#     color_scheme = st.selectbox("Color Scheme", ["Plotly Express", "Streamlit Default", "Custom"])
#     st.write(f"Selected Color Scheme: {color_scheme}")
#
#     # Example: Adjusting plot size
#     plot_size = st.slider("Plot Size", min_value=200, max_value=1000, value=500, step=50)
#     st.write(f"Selected Plot Size: {plot_size}")
#
#     st.header("Data Filtering Defaults")
#     # Example: Default year selection
#     default_year = st.selectbox("Default Year", df['Year'].unique())
#     st.write(f"Selected Default Year: {default_year}")
#
#
# # Run the main function
# if __name__ == "__main__":
#     main()

# import streamlit as st
# import pandas as pd
# import numpy as np
# from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
# from sklearn.preprocessing import StandardScaler
# import plotly.express as px
# from PIL import Image
# import base64
# from io import BytesIO  # Import BytesIO for handling bytes data
#
# # Load and preprocess data
# @st.cache_data
# def load_data():
#     # Load the data (replace with your data loading logic)
#     df = pd.read_csv("uncleaned_traffic.csv")
#
#     # Your data preprocessing steps go here
#     # Data cleaning
#     df = df.dropna(subset=['DateTime', 'Junction', 'Vehicles', 'ID'])
#     df = df.drop_duplicates()
#     df['DateTime'] = pd.to_datetime(df['DateTime'])
#     df['Vehicles'] = pd.to_numeric(df['Vehicles'], errors='coerce')  # Ensure 'Vehicles' column is numeric
#     df = df.dropna(subset=['Vehicles'])
#     df['Junction'] = df['Junction'].astype(int)
#     df['ID'] = df['ID'].apply(lambda x: '{:.0f}'.format(x))
#     df = df.drop_duplicates()
#     df = df.sort_values(by=['Junction', 'DateTime'])
#     df.reset_index(drop=True, inplace=True)
#     df.set_index('DateTime', inplace=True)
#
#     # Handling missing values
#     df.fillna(method='ffill', inplace=True)
#
#     # Extracting date-time features
#     df['Year'] = df.index.year
#     df['Month'] = df.index.month
#     df['Day'] = df.index.day
#     df['Hour'] = df.index.hour
#     df['DayOfWeek'] = df.index.dayofweek
#
#     # Adding cyclical features
#     df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
#     df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
#     df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
#     df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
#
#     # Adding holiday feature
#     cal = calendar()
#     holidays = cal.holidays(start=df.index.min(), end=df.index.max())
#     df['Holiday'] = df.index.isin(holidays).astype(int)
#
#     # Feature scaling
#     scaler = StandardScaler()
#     df['Vehicles_scaled'] = scaler.fit_transform(df[['Vehicles']])
#
#     return df
#
#
# # Initialize session state
# if 'uploaded_photos' not in st.session_state:
#     st.session_state.uploaded_photos = []
#
#
# # Main function to run the Streamlit app
# def main():
#     # Set page title and icon
#     st.set_page_config(page_title="Traffic Data Analysis", page_icon=":car:")
#
#     # Load data
#     df = load_data()
#
#     # # Create header with title and logos
#     # st.title("Traffic Data Analysis")
#
#     # Layout for logos in the corners
#     col1, col2, _ = st.columns([10,1,1])
#
#     with col1:
#         st.image("traffic-police.jpg", width=150)
#
#     with col2:
#         st.image("logo.jpg", width=150)
#
#     # Create sidebar navigation
#     st.sidebar.title("Navigation")
#     selected_tab = st.sidebar.radio("Go to", ["Traffic Data", "Upload Photos", "Settings"])
#
#     # Display selected tab content
#     if selected_tab == "Traffic Data":
#         display_traffic_data(df)
#     elif selected_tab == "Upload Photos":
#         display_upload_photos()
#     elif selected_tab == "Settings":
#         display_settings(df)
#
#
# # Function to display Traffic Data tab
# def display_traffic_data(df):
#     st.title("Traffic Data Analysis")
#
#     # Check if 'Year' column exists in the DataFrame
#     if 'Year' not in df.columns:
#         st.error("Year column not found in the DataFrame!")
#         return
#
#     # Initialize junctions variable
#     junctions = []
#
#     # Sidebar filters
#     st.sidebar.title("Traffic Data Filters")
#
#     # Checkbox for selecting specific date, year, and hour
#     show_date_checkbox = st.sidebar.checkbox("Filter by Date")
#
#     if show_date_checkbox:
#         st.sidebar.title("Date Filters")
#         year = st.sidebar.selectbox("Select Year", np.append(["Overall"], df['Year'].unique()))
#         month = st.sidebar.selectbox("Select Month", np.append(["Overall"], df['Month'].unique()))
#         day = st.sidebar.selectbox("Select Day", np.append(["Overall"], df['Day'].unique()))
#
#         if year == "Overall":
#             filtered_df = df
#         else:
#             filtered_df = df[df['Year'] == int(year)]
#
#         if month != "Overall":
#             filtered_df = filtered_df[filtered_df['Month'] == int(month)]
#
#         if day != "Overall":
#             filtered_df = filtered_df[filtered_df['Day'] == int(day)]
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}")
#
#     else:
#         st.sidebar.title("Date and Time Filters")
#         year = st.sidebar.selectbox("Select Year", np.append(["Overall"], df['Year'].unique()))
#         month = st.sidebar.selectbox("Select Month", np.append(["Overall"], df['Month'].unique()))
#         day = st.sidebar.selectbox("Select Day", np.append(["Overall"], df['Day'].unique()))
#         hour = st.sidebar.selectbox("Select Hour", np.append(["Overall"], df['Hour'].unique()))
#         junctions = st.sidebar.multiselect("Select Junction", df['Junction'].unique(), default=df['Junction'].unique())
#
#         filtered_df = df
#
#         if year != "Overall":
#             filtered_df = filtered_df[filtered_df['Year'] == int(year)]
#
#         if month != "Overall":
#             filtered_df = filtered_df[filtered_df['Month'] == int(month)]
#
#         if day != "Overall":
#             filtered_df = filtered_df[filtered_df['Day'] == int(day)]
#
#         if hour != "Overall":
#             filtered_df = filtered_df[filtered_df['Hour'] == int(hour)]
#
#         if junctions:
#             filtered_df = filtered_df[filtered_df['Junction'].isin(junctions)]
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}, Hour: {hour}, Junctions: {junctions}")
#
#     # Display filtered data
#     st.write(filtered_df)
#
#     # Visualization of filtered data
#     st.title("Historical Traffic Data - Line Chart")
#
#     if not filtered_df.empty:
#         # Group by Hour and Junction to get the average number of vehicles
#         historical_df = filtered_df.groupby(['Hour', 'Junction'], as_index=False)['Vehicles'].mean()
#
#         # Create line chart using Plotly Express
#         fig = px.line(historical_df, x='Hour', y='Vehicles', color='Junction',
#                       title='Average Hourly Traffic Data',
#                       labels={'Hour': 'Hour of Day', 'Vehicles': 'Average Number of Vehicles'})
#         st.plotly_chart(fig)
#
#
# # Function to display Upload Photos tab
# def display_upload_photos():
#     st.title("Upload Photos")
#
#     # Allow users to upload photos
#     uploaded_files = st.file_uploader("Choose a photo", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
#
#     # Display uploaded photos
#     if uploaded_files:
#         for uploaded_file in uploaded_files:
#             try:
#                 # Open and resize the image
#                 img = Image.open(uploaded_file)
#                 img = img.resize((200, 200))  # Adjust the size as needed
#                 st.image(img, caption=uploaded_file.name)
#
#                 # Append uploaded files to session state
#                 st.session_state.uploaded_photos.append(uploaded_file)
#
#             except UnidentifiedImageError as e:
#                 st.warning(f"Unable to identify image file {uploaded_file.name}: {e}")
#
#         # Display success message
#         st.success("Photos uploaded successfully!")
#
#     # Display option to download uploaded photos
#     if st.session_state.uploaded_photos:
#         st.title("Download Uploaded Photos")
#
#         # Display links for downloading uploaded photos
#         for i, uploaded_file in enumerate(st.session_state.uploaded_photos, start=1):
#             photo_name = uploaded_file.name
#             st.markdown(f"**{photo_name}**: ")
#             st.markdown(get_image_download_link(uploaded_file, photo_name), unsafe_allow_html=True)
#
#
# # Function to get download link for an image
# def get_image_download_link(img, text):
#     img_read = img.read()  # Read the file data
#     img = Image.open(img)  # Open the image with PIL (already opened in display_upload_photos)
#
#     # Resize the image to a smaller size (optional)
#     img = img.resize((200, 200))
#
#     # Save the image to a BytesIO buffer as JPEG
#     img_buffer = BytesIO()
#     img.save(img_buffer, format="JPEG")
#     img_str = base64.b64encode(img_buffer.getvalue()).decode()
#
#     # Generate download link
#     href = f'<a href="data:file/jpg;base64,{img_str}" download="{text}.jpg">Download {text}</a>'
#     return href
#
#
# # Function to display Settings tab
# def display_settings(df):
#     st.title("Settings")
#
#     st.header("Visualization Settings")
#     # Example: Adjusting line chart granularity
#     line_chart_granularity = st.selectbox("Line Chart Granularity", ["Hour", "Day", "Month"])
#     st.write(f"Selected Line Chart Granularity: {line_chart_granularity}")
#
#     # Example: Adjusting color scheme
#     color_scheme = st.selectbox("Color Scheme", ["Plotly Express", "Streamlit Default", "Custom"])
#     st.write(f"Selected Color Scheme: {color_scheme}")
#
#     # Example: Adjusting plot size
#     plot_size = st.slider("Plot Size", min_value=200, max_value=1000, value=500, step=50)
#     st.write(f"Selected Plot Size: {plot_size}")
#
#     st.header("Data Filtering Defaults")
#     # Example: Default year selection
#     default_year = st.selectbox("Default Year", df['Year'].unique())
#     st.write(f"Selected Default Year: {default_year}")
#
#
# # Run the main function
# if __name__ == "__main__":
#     main()
# import streamlit as st
# import pandas as pd
# import numpy as np
# from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
# from sklearn.preprocessing import StandardScaler
# import plotly.express as px
# from PIL import Image, UnidentifiedImageError
# import base64
# from io import BytesIO  # Import BytesIO for handling bytes data
#
# # Load and preprocess data
# @st.cache_data
# def load_data():
#     # Load the data (replace with your data loading logic)
#     df = pd.read_csv("uncleaned_traffic.csv")
#
#     # Your data preprocessing steps go here
#     # Data cleaning
#     df = df.dropna(subset=['DateTime', 'Junction', 'Vehicles', 'ID'])
#     df = df.drop_duplicates()
#     df['DateTime'] = pd.to_datetime(df['DateTime'])
#     df['Vehicles'] = pd.to_numeric(df['Vehicles'], errors='coerce')  # Ensure 'Vehicles' column is numeric
#     df = df.dropna(subset=['Vehicles'])
#     df['Junction'] = df['Junction'].astype(int)
#     df['ID'] = df['ID'].apply(lambda x: '{:.0f}'.format(x))
#     df = df.drop_duplicates()
#     df = df.sort_values(by=['Junction', 'DateTime'])
#     df.reset_index(drop=True, inplace=True)
#     df.set_index('DateTime', inplace=True)
#
#     # Handling missing values
#     df.fillna(method='ffill', inplace=True)
#
#     # Extracting date-time features
#     df['Year'] = df.index.year
#     df['Month'] = df.index.month
#     df['Day'] = df.index.day
#     df['Hour'] = df.index.hour
#     df['DayOfWeek'] = df.index.dayofweek
#
#     # Adding cyclical features
#     df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
#     df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
#     df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
#     df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
#
#     # Adding holiday feature
#     cal = calendar()
#     holidays = cal.holidays(start=df.index.min(), end=df.index.max())
#     df['Holiday'] = df.index.isin(holidays).astype(int)
#
#     # Feature scaling
#     scaler = StandardScaler()
#     df['Vehicles_scaled'] = scaler.fit_transform(df[['Vehicles']])
#
#     return df
#
#
# # Initialize session state
# if 'uploaded_photos' not in st.session_state:
#     st.session_state.uploaded_photos = pd.DataFrame(columns=['Photo Name', 'Download Link'])
#
#
# # Main function to run the Streamlit app
# def main():
#     # Set page title and icon
#     st.set_page_config(page_title="Traffic Data Analysis", page_icon=":car:")
#
#     # Load data
#     df = load_data()
#
#     # Layout for logos in the corners
#     col1, col2, _ = st.columns([10, 1, 1])
#
#     with col1:
#         st.image("traffic-police.jpg", width=150)
#
#     with col2:
#         st.image("logo.jpg", width=150)
#
#     # Create sidebar navigation
#     st.sidebar.title("Navigation")
#     selected_tab = st.sidebar.radio("Go to", ["Traffic Data", "Upload Photos", "Settings"])
#
#     # Display selected tab content
#     if selected_tab == "Traffic Data":
#         display_traffic_data(df)
#     elif selected_tab == "Upload Photos":
#         display_upload_photos()
#     elif selected_tab == "Settings":
#         display_settings(df)
#
#
# # Function to display Traffic Data tab
# def display_traffic_data(df):
#     st.title("Traffic Data Analysis")
#
#     # Check if 'Year' column exists in the DataFrame
#     if 'Year' not in df.columns:
#         st.error("Year column not found in the DataFrame!")
#         return
#
#     # Initialize junctions variable
#     junctions = []
#
#     # Sidebar filters
#     st.sidebar.title("Traffic Data Filters")
#
#     # Checkbox for selecting specific date, year, and hour
#     show_date_checkbox = st.sidebar.checkbox("Filter by Date")
#
#     if show_date_checkbox:
#         st.sidebar.title("Date Filters")
#         year = st.sidebar.selectbox("Select Year", np.append(["Overall"], df['Year'].unique()))
#         month = st.sidebar.selectbox("Select Month", np.append(["Overall"], df['Month'].unique()))
#         day = st.sidebar.selectbox("Select Day", np.append(["Overall"], df['Day'].unique()))
#
#         if year == "Overall":
#             filtered_df = df
#         else:
#             filtered_df = df[df['Year'] == int(year)]
#
#         if month != "Overall":
#             filtered_df = filtered_df[filtered_df['Month'] == int(month)]
#
#         if day != "Overall":
#             filtered_df = filtered_df[filtered_df['Day'] == int(day)]
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}")
#
#     else:
#         st.sidebar.title("Date and Time Filters")
#         year = st.sidebar.selectbox("Select Year", np.append(["Overall"], df['Year'].unique()))
#         month = st.sidebar.selectbox("Select Month", np.append(["Overall"], df['Month'].unique()))
#         day = st.sidebar.selectbox("Select Day", np.append(["Overall"], df['Day'].unique()))
#         hour = st.sidebar.selectbox("Select Hour", np.append(["Overall"], df['Hour'].unique()))
#         junctions = st.sidebar.multiselect("Select Junction", df['Junction'].unique(), default=df['Junction'].unique())
#
#         filtered_df = df
#
#         if year != "Overall":
#             filtered_df = filtered_df[filtered_df['Year'] == int(year)]
#
#         if month != "Overall":
#             filtered_df = filtered_df[filtered_df['Month'] == int(month)]
#
#         if day != "Overall":
#             filtered_df = filtered_df[filtered_df['Day'] == int(day)]
#
#         if hour != "Overall":
#             filtered_df = filtered_df[filtered_df['Hour'] == int(hour)]
#
#         if junctions:
#             filtered_df = filtered_df[filtered_df['Junction'].isin(junctions)]
#
#         st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}, Hour: {hour}, Junctions: {junctions}")
#
#     # Display filtered data
#     st.write(filtered_df)
#
#     # Visualization of filtered data
#     st.title("Historical Traffic Data - Line Chart")
#
#     if not filtered_df.empty:
#         # Group by Hour and Junction to get the average number of vehicles
#         historical_df = filtered_df.groupby(['Hour', 'Junction'], as_index=False)['Vehicles'].mean()
#
#         # Create line chart using Plotly Express
#         fig = px.line(historical_df, x='Hour', y='Vehicles', color='Junction',
#                       title='Average Hourly Traffic Data',
#                       labels={'Hour': 'Hour of Day', 'Vehicles': 'Average Number of Vehicles'})
#         st.plotly_chart(fig)
#
#
# # Function to display Upload Photos tab
# def display_upload_photos():
#     st.title("Upload Photos")
#
#     # Allow users to upload photos
#     uploaded_files = st.file_uploader("Choose a photo", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
#
#     # Display uploaded photos
#     if uploaded_files:
#         for uploaded_file in uploaded_files:
#             try:
#                 # Open and resize the image
#                 img = Image.open(uploaded_file)
#                 img = img.resize((200, 200))  # Adjust the size as needed
#                 st.image(img, caption=uploaded_file.name)
#
#                 # Append uploaded files to session state
#                 st.session_state.uploaded_photos = pd.concat([
#                     st.session_state.uploaded_photos,
#                     pd.DataFrame([{
#                         'Photo Name': uploaded_file.name,
#                         'Download Link': get_image_download_link(uploaded_file, uploaded_file.name)
#                     }])
#                 ], ignore_index=True)
#
#             except UnidentifiedImageError as e:
#                 st.warning(f"Unable to identify image file {uploaded_file.name}: {e}")
#
#         # Save updated uploaded photos to CSV
#         save_uploaded_photos_to_csv()
#
#         # Display success message
#         st.success("Photos uploaded successfully!")
#
#     # Display option to download uploaded photos
#     if not st.session_state.uploaded_photos.empty:
#         st.title("Download Uploaded Photos")
#
#         # Display links for downloading uploaded photos
#         for i, row in st.session_state.uploaded_photos.iterrows():
#             st.markdown(f"**{row['Photo Name']}**: ")
#             st.markdown(row['Download Link'], unsafe_allow_html=True)
#
#
# # Function to save uploaded photos DataFrame to CSV
# def save_uploaded_photos_to_csv():
#     st.session_state.uploaded_photos.to_csv("uploaded_photos.csv", index=False)
#
#
# # Function to get download link for an image
# def get_image_download_link(img, text):
#     img_read = img.read()  # Read the file data
#     img = Image.open(img)  # Open the image with PIL (already opened in display_upload_photos)
#
#     # Resize the image to a smaller size (optional)
#     img = img.resize((200, 200))
#
#     # Save the image to a BytesIO buffer as JPEG
#     img_buffer = BytesIO()
#     img.save(img_buffer, format="JPEG")
#     img_str = base64.b64encode(img_buffer.getvalue()).decode()
#
#     # Generate download link
#     href = f'<a href="data:file/jpg;base64,{img_str}" download="{text}.jpg">Download {text}</a>'
#     return href
#
#
# # Function to display Settings tab
# def display_settings(df):
#     st.title("Settings")
#
#     st.header("Visualization Settings")
#     # Example: Adjusting line chart granularity
#     line_chart_granularity = st.selectbox("Line Chart Granularity", ["Hour", "Day", "Month"])
#     st.write(f"Selected Line Chart Granularity: {line_chart_granularity}")
#
#     # Example: Adjusting color scheme
#     color_scheme = st.selectbox("Color Scheme", ["Plotly Express", "Streamlit Default", "Custom"])
#     st.write(f"Selected Color Scheme: {color_scheme}")
#
#     # Example: Adjusting plot size
#     plot_size = st.slider("Plot Size", min_value=200, max_value=1000, value=500, step=50)
#     st.write(f"Selected Plot Size: {plot_size}")
#
#     st.header("Data Filtering Defaults")
#     # Example: Default year selection
#     default_year = st.selectbox("Default Year", df['Year'].unique())
#     st.write(f"Selected Default Year: {default_year}")
#
#
# # Run the main function
# if __name__ == "__main__":
#     main()
#
import streamlit as st
import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from PIL import Image, UnidentifiedImageError
import base64
from io import BytesIO  # Import BytesIO for handling bytes data

# Load and preprocess data
@st.cache_data
def load_data():
    # Load the data (replace with your data loading logic)
    df = pd.read_csv("uncleaned_traffic.csv")

    # Your data preprocessing steps go here
    # Data cleaning
    df = df.dropna(subset=['DateTime', 'Junction', 'Vehicles', 'ID'])
    df = df.drop_duplicates()
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['Vehicles'] = pd.to_numeric(df['Vehicles'], errors='coerce')  # Ensure 'Vehicles' column is numeric
    df = df.dropna(subset=['Vehicles'])
    df['Junction'] = df['Junction'].astype(int)
    df['ID'] = df['ID'].apply(lambda x: '{:.0f}'.format(x))
    df = df.drop_duplicates()
    df = df.sort_values(by=['Junction', 'DateTime'])
    df.reset_index(drop=True, inplace=True)
    df.set_index('DateTime', inplace=True)

    # Handling missing values
    df.fillna(method='ffill', inplace=True)

    # Extracting date-time features
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Day'] = df.index.day
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek

    # Adding cyclical features
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)

    # Adding holiday feature
    cal = calendar()
    holidays = cal.holidays(start=df.index.min(), end=df.index.max())
    df['Holiday'] = df.index.isin(holidays).astype(int)

    # Feature scaling
    scaler = StandardScaler()
    df['Vehicles_scaled'] = scaler.fit_transform(df[['Vehicles']])

    return df


# Initialize session state
if 'uploaded_photos' not in st.session_state:
    st.session_state.uploaded_photos = pd.DataFrame(columns=['Photo Name', 'Download Link', 'Image'])


# Main function to run the Streamlit app
def main():
    # Set page title and icon
    st.set_page_config(page_title="Traffic Data Analysis", page_icon=":car:")

    # Load data
    df = load_data()

    # Layout for logos in the corners
    col1, col2, _ = st.columns([10, 1, 1])

    with col1:
        st.image("traffic-police.jpg", width=150)

    with col2:
        st.image("logo.jpg", width=150)

    # Create sidebar navigation
    st.sidebar.title("Navigation")
    selected_tab = st.sidebar.radio("Go to", ["Traffic Data", "Upload Photos", "Settings"])

    # Display selected tab content
    if selected_tab == "Traffic Data":
        display_traffic_data(df)
    elif selected_tab == "Upload Photos":
        display_upload_photos()
    elif selected_tab == "Settings":
        display_settings(df)


# Function to display Traffic Data tab
def display_traffic_data(df):
    st.title("Traffic Data Analysis")

    # Check if 'Year' column exists in the DataFrame
    if 'Year' not in df.columns:
        st.error("Year column not found in the DataFrame!")
        return

    # Initialize junctions variable
    junctions = []

    # Sidebar filters
    st.sidebar.title("Traffic Data Filters")

    # Checkbox for selecting specific date, year, and hour
    show_date_checkbox = st.sidebar.checkbox("Filter by Date")

    if show_date_checkbox:
        st.sidebar.title("Date Filters")
        year = st.sidebar.selectbox("Select Year", np.append(["Overall"], df['Year'].unique()))
        month = st.sidebar.selectbox("Select Month", np.append(["Overall"], df['Month'].unique()))
        day = st.sidebar.selectbox("Select Day", np.append(["Overall"], df['Day'].unique()))

        if year == "Overall":
            filtered_df = df
        else:
            filtered_df = df[df['Year'] == int(year)]

        if month != "Overall":
            filtered_df = filtered_df[filtered_df['Month'] == int(month)]

        if day != "Overall":
            filtered_df = filtered_df[filtered_df['Day'] == int(day)]

        st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}")

    else:
        st.sidebar.title("Date and Time Filters")
        year = st.sidebar.selectbox("Select Year", np.append(["Overall"], df['Year'].unique()))
        month = st.sidebar.selectbox("Select Month", np.append(["Overall"], df['Month'].unique()))
        day = st.sidebar.selectbox("Select Day", np.append(["Overall"], df['Day'].unique()))
        hour = st.sidebar.selectbox("Select Hour", np.append(["Overall"], df['Hour'].unique()))
        junctions = st.sidebar.multiselect("Select Junction", df['Junction'].unique(), default=df['Junction'].unique())

        filtered_df = df

        if year != "Overall":
            filtered_df = filtered_df[filtered_df['Year'] == int(year)]

        if month != "Overall":
            filtered_df = filtered_df[filtered_df['Month'] == int(month)]

        if day != "Overall":
            filtered_df = filtered_df[filtered_df['Day'] == int(day)]

        if hour != "Overall":
            filtered_df = filtered_df[filtered_df['Hour'] == int(hour)]

        if junctions:
            filtered_df = filtered_df[filtered_df['Junction'].isin(junctions)]

        st.write(f"Showing data for Year: {year}, Month: {month}, Day: {day}, Hour: {hour}, Junctions: {junctions}")

    # Display filtered data
    st.write(filtered_df)

    # Visualization of filtered data
    st.title("Historical Traffic Data - Line Chart")

    if not filtered_df.empty:
        # Group by Hour and Junction to get the average number of vehicles
        historical_df = filtered_df.groupby(['Hour', 'Junction'], as_index=False)['Vehicles'].mean()

        # Create line chart using Plotly Express
        fig = px.line(historical_df, x='Hour', y='Vehicles', color='Junction',
                      title='Average Hourly Traffic Data',
                      labels={'Hour': 'Hour of Day', 'Vehicles': 'Average Number of Vehicles'})
        st.plotly_chart(fig)


# Function to display Upload Photos tab
def display_upload_photos():
    st.title("Upload Photos")

    # Allow users to upload photos
    uploaded_files = st.file_uploader("Choose a photo", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

    # Display uploaded photos
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                # Open and resize the image
                img = Image.open(uploaded_file)
                img = img.resize((200, 200))  # Adjust the size as needed
                st.image(img, caption=uploaded_file.name)

                # Check if the uploaded file already exists in session state
                existing_index = st.session_state.uploaded_photos[
                    st.session_state.uploaded_photos['Photo Name'] == uploaded_file.name].index

                if len(existing_index) > 0:
                    # Update existing image
                    st.session_state.uploaded_photos.at[existing_index[0], 'Image'] = img
                else:
                    # Append uploaded files to session state
                    st.session_state.uploaded_photos = pd.concat([
                        st.session_state.uploaded_photos,
                        pd.DataFrame([{
                            'Photo Name': uploaded_file.name,
                            'Download Link': get_image_download_link(uploaded_file, uploaded_file.name),
                            'Image': img
                        }])
                    ], ignore_index=True)

            except UnidentifiedImageError as e:
                st.warning(f"Unable to identify image file {uploaded_file.name}: {e}")

        # Save updated uploaded photos to CSV
        save_uploaded_photos_to_csv()

        # Display success message
        st.success("Photos uploaded successfully!")

    # Display option to delete uploaded photos
    if not st.session_state.uploaded_photos.empty:
        st.title("Manage Uploaded Photos")

        # Display uploaded photos with delete button
        for i, row in st.session_state.uploaded_photos.iterrows():
            st.image(row['Image'], caption=row['Photo Name'], width=200)

            # Add delete button
            if st.button(f"Delete {row['Photo Name']}"):
                st.session_state.uploaded_photos = st.session_state.uploaded_photos.drop(index=i).reset_index(drop=True)
                st.success(f"Deleted {row['Photo Name']}")

                # Save updated uploaded photos to CSV
                save_uploaded_photos_to_csv()


# Function to save uploaded photos to CSV
def save_uploaded_photos_to_csv():
    st.session_state.uploaded_photos.to_csv("uploaded_photos.csv", index=False)


# Function to get download link for an image
def get_image_download_link(img, text):
    img_read = img.read()  # Read the file data
    img = Image.open(img)  # Open the image with PIL (already opened in display_upload_photos)

    # Resize the image to a smaller size (optional)
    img = img.resize((200, 200))

    # Save the image to a BytesIO buffer as JPEG
    img_buffer = BytesIO()
    img.save(img_buffer, format="JPEG")
    img_str = base64.b64encode(img_buffer.getvalue()).decode()

    # Generate download link
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{text}.jpg">Download {text}</a>'
    return href


# Function to display Settings tab
def display_settings(df):
    st.title("Settings")

    st.header("Visualization Settings")
    # Example: Adjusting line chart granularity
    line_chart_granularity = st.selectbox("Line Chart Granularity", ["Hour", "Day", "Month"])
    st.write(f"Selected Line Chart Granularity: {line_chart_granularity}")

    # Example: Adjusting color scheme
    color_scheme = st.selectbox("Color Scheme", ["Plotly Express", "Streamlit Default", "Custom"])
    st.write(f"Selected Color Scheme: {color_scheme}")

    # Example: Adjusting plot size
    plot_size = st.slider("Plot Size", min_value=200, max_value=1000, value=500, step=50)
    st.write(f"Selected Plot Size: {plot_size}")

    st.header("Data Filtering Defaults")
    # Example: Default year selection
    default_year = st.selectbox("Default Year", df['Year'].unique())
    st.write(f"Selected Default Year: {default_year}")


# Run the main function
if __name__ == "__main__":
    main()
