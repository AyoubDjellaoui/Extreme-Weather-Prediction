import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import folium
from folium import PolyLine
import streamlit.components.v1 as components

st.header("Exploratory Data Analysis (EDA)")

st.markdown("""
Nous avons téléchargé le jeu de données sur les tornades couvrant la période de 1950 à 2022 depuis le site web du National Weather Service (NWS) Storm Prediction Center (SPC). Ce jeu de données comprend des rapports détaillés sur les tornades, tels que la date, l'heure, la magnitude, les pertes, la largeur, la longueur, les points de départ et d'arrivée.

- **Lien de téléchargement**: [NOAA SPC Tornado Data](https://www.spc.noaa.gov/wcm/#data)
- **Lien de la documentation**: [SPC Severe Weather Database Description](https://www.spc.noaa.gov/wcm/data/SPC_severe_database_description.pdf)
- **Description du dataset**: Le dataset contient 68,693 entrées avec 27 colonnes, y compris om, yr, mo, dy, date, time, tz, datetime_utc, st, stf, mag, inj, fat, loss, slat, slon, elat, elon, len, wid
""")

# Load data
def load_data(loc):
    return pd.read_csv(
        loc,
        parse_dates=['date'],
        dtype={'st':'category'},
    ).assign(
        elat=lambda d: d.elat.mask(d.elat == 0.0, d.slat),
        elon=lambda d: d.elon.mask(d.elon == 0.0, d.slon),
        mag=lambda d: d.mag.mask(d.mag == -9, 0),
    )

data_loc = 'data/Tornados 1950 - 2022.csv'
df = load_data(data_loc)

st.subheader("Données sur les Tornades (1950-2022)")
st.write(df.head())

# Summary Statistics
st.subheader("Summary Statistics")
st.write(df.describe())

# Tornades par Année
st.subheader("Tornades par Année")
st.write("Cette section présente le nombre de tornades enregistrées chaque année entre 1950 et 2022.")
data = df['yr'].value_counts().to_frame().reset_index()
data.columns = ['yr', 'count']
fig = px.bar(data, x='yr', y='count', title='Tornades par Année')
fig.update_layout(
    title={'text': "Tornades par Année", 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 30}},
    xaxis_title='Année', yaxis_title='Nombre', template='plotly_white', font=dict(size=18)
)
fig.update_traces(marker_color='#50777b')
st.plotly_chart(fig)

# Tornades par Année avec Ligne de Tendance
fig = px.scatter(data, x='yr', y='count', trendline='ols', title='Tornades par Année avec Ligne de Tendance')
fig.update_layout(
    title={'text': "Tornades par Année avec Ligne de Tendance", 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 30}},
    xaxis_title='Année', yaxis_title='Nombre', template='plotly_white', font=dict(size=18)
)
fig.update_traces(marker_color='#50777b')
st.plotly_chart(fig)

# Tornades par Mois
st.subheader("Tornades par Mois")
st.write("Cette section montre le nombre de tornades par mois, ce qui nous aide à comprendre la saisonnalité des tornades.")
fig = px.histogram(df, x='mo', title='Tornades par Mois')
fig.update_layout(
    title={'text': "Tornades par Mois", 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 30}},
    xaxis_title='Mois', yaxis_title='Nombre', template='plotly_white', font=dict(size=18)
)
fig.update_traces(marker_color='#50777b')
st.plotly_chart(fig)

# Magnitude des Tornades
st.subheader("Magnitude des Tornades")
st.write("Cette section illustre la répartition des tornades par magnitude, nous aidant à comprendre la fréquence des tornades plus fortes.")
fig = px.histogram(df, x='mag', title='Tendance des Tornades par Magnitude')
fig.update_layout(
    title={'text': "Tendance des Tornades par Magnitude", 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 30}},
    xaxis_title='Magnitude', yaxis_title='Nombre', template='plotly_white', font=dict(size=18)
)
fig.update_traces(marker_color='#50777b')
st.plotly_chart(fig)

# Magnitudes des Tornades par Année
st.subheader("Magnitudes des Tornades par Année")
st.write("""
À première vue, les tornades montrent une forte tendance à la hausse lorsque la magnitude est de 0 et 1 jusqu'au début des années 2000, une légère tendance à la baisse lorsque la magnitude est de 2, et aucun changement significatif lorsque la magnitude est de 3, 4, 5.

Il semble qu'il y ait une tendance à la baisse du nombre de tornades de catégorie plus forte et une tendance à la hausse du nombre de tornades de catégorie plus faible. Cependant, l'article Wikipédia sur le "Radar météorologique" indique qu'entre 1980 et 2000, les réseaux de radars météorologiques sont devenus la norme avec le remplacement des radars conventionnels par des radars Doppler. Ce changement a entraîné une augmentation de la capacité de détection, notamment des tempêtes plus faibles, ce qui pourrait fausser la tendance à la hausse.
""")
to = df.groupby(['yr', 'mag'])['yr'].count().reset_index(name='Count')
plt.figure(figsize=(15, 5))
sns.lineplot(data=to, x='yr', y='Count', hue='mag', marker='o', palette='viridis').set(title='Magnitudes des Tornades par Année')
plt.xlabel('Année')
plt.ylabel('Nombre')

# Save the plot with higher resolution
st.pyplot(plt)
plt.close()

# Analyse Spatiale
st.subheader("Analyse Spatiale")
st.write("Cette section montre la répartition géographique des tornades aux États-Unis.")
fig = px.scatter_geo(df, lat='slat', lon='slon', color='mag', width=800, height=400, color_continuous_scale=px.colors.sequential.Aggrnyl)
fig.update_layout(
    title={'text': 'Occurrences de Tornades aux États-Unis (1950-2022)', 'x':0.5, 'xanchor':'center', 'yanchor':'top', 'font':{'size':24}},
    geo_scope='usa'
)
fig.update_traces(marker=dict(size=1), showlegend=False)
st.plotly_chart(fig)

# Top 5 États par Nombre de Blessures (1950-2022)
inj_state = df.st.value_counts().head(5).reset_index(name='counts')
inj_state.columns = ['State', 'Counts']
inj_state = inj_state.sort_values(by='Counts', ascending=True)  # Sort in ascending order for top-down bars

fig = px.bar(inj_state, y='State', x='Counts', title='Top 5 États par Nombre de Blessures (1950-2022)', template='simple_white', orientation='h')
fig.update_layout(
    title={'text': 'Top 5 États par Nombre de Blessures (1950-2022)', 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 24}},
    xaxis_title='Nombre de Blessures', yaxis_title='État', font=dict(size=18), bargap=0.1  # Reduce gap between bars
)
fig.update_traces(marker_color='#50777b')
st.plotly_chart(fig)

# Top 5 États par Nombre de Décès (1950-2022)
fat_state = df.groupby(['st']).sum(numeric_only=True)['fat'].reset_index(name='counts').sort_values('counts', ascending=False).head(5)
fat_state.columns = ['State', 'Counts']
fat_state = fat_state.sort_values(by='Counts', ascending=True)  # Sort in ascending order for top-down bars

fig = px.bar(fat_state, y='State', x='Counts', title='Top 5 États par Nombre de Décès (1950-2022)', template='simple_white', orientation='h')
fig.update_layout(
    title={'text': 'Top 5 États par Nombre de Décès (1950-2022)', 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 24}},
    xaxis_title='Nombre de Décès', yaxis_title='État', font=dict(size=18), bargap=0.1  # Reduce gap between bars
)
fig.update_traces(marker_color='#50777b')
st.plotly_chart(fig)

# Top 5 États par Pertes Financières (1950-2022)
loss_state = df.groupby(['st']).sum(numeric_only=True)['loss'].reset_index(name='counts').sort_values('counts', ascending=False).head(5)
loss_state.columns = ['State', 'Counts']
loss_state = loss_state.sort_values(by='Counts', ascending=True)  # Sort in ascending order for top-down bars

fig = px.bar(loss_state, y='State', x='Counts', title='Top 5 États par Pertes Financières (1950-2022)', template='simple_white', orientation='h')
fig.update_layout(
    title={'text': 'Top 5 États par Pertes Financières (1950-2022)', 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 24}},
    xaxis_title='Billion USD', yaxis_title='État', font=dict(size=18), bargap=0.1  # Reduce gap between bars
)
fig.update_traces(marker_color='#50777b')
fig.update_layout(xaxis=dict(tickformat=",.0fB"))
st.plotly_chart(fig)

# Longueur des Tornades
st.subheader("Longueur des Tornades")
st.write("Cette section présente les 10 tornades les plus longues enregistrées.")
top_10_leng = df.nlargest(10, 'len')
fig = px.bar(top_10_leng, x='len', y='st', hover_data=['mo', 'yr', 'fat'], color='len', labels={'yr':'Année', 'mo':'Mois', 'len':'Longueur de la Tornade', 'st':'État', 'fat':'Décès'})
fig.update_layout(title={'text': 'Top 10 Tornades par Longueur', 'x':0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 24}})
fig.update_traces(marker_color='#50777b')
st.plotly_chart(fig)

# Occurrences par Heure
st.subheader("Occurrences par Heure")
st.write("Cette section montre le nombre de tornades par heure de la journée.")
t = df.groupby(['time'])['time'].count().reset_index(name='count')
fig = px.line(t, x='time', y='count', title='Occurrences de Tornades par Heure de la Journée', template='simple_white')
fig.update_traces(line_color='#50777b', line_width=2)
fig.update_layout(
    title={'text': 'Occurrences de Tornades par Heure de la Journée', 'x':0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 24}},
    xaxis_title='Heure de la Journée', yaxis_title='Nombre', font=dict(size=18)
)
st.plotly_chart(fig)

# Tornades par État
st.subheader("Tornades par État")
st.write("Cette section présente le nombre de tornades par état de 1950 à 2022.")
state_counts = df['st'].value_counts().reset_index()
state_counts.columns = ['State', 'Count']
fig = px.bar(state_counts, x='State', y='Count', title='Nombre de Tornades par État (1950-2022)', template='simple_white')
fig.update_layout(
    title={'text': 'Nombre de Tornades par État (1950-2022)', 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 24}},
    xaxis_title="Code de l'État", yaxis_title='Nombre de Tornades', font=dict(size=14)
)
fig.update_traces(marker_color='#50777b')
st.plotly_chart(fig)

# Strongest Tornadoes by State
st.subheader("Tornades de Magnitude 4-5 par État")
st.write("Cette section présente le nombre de tornades de magnitude 4-5 par état.")
strongest = df[(df['mag'] == 5.0) | (df['mag'] == 4.0)]
strongest_counts = strongest['st'].value_counts().reset_index()
strongest_counts.columns = ['State', 'Count']
fig = px.bar(strongest_counts, x='State', y='Count', title='Nombre de Tornades de Magnitude 4-5 par État', template='simple_white')
fig.update_layout(
    title={'text': 'Nombre de Tornades de Magnitude 4-5 par État', 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 24}},
    xaxis_title="Code de l'État", yaxis_title='Nombre de Tornades', font=dict(size=18)
)
fig.update_traces(marker_color='#50777b')
st.plotly_chart(fig)

# Chemins des Tornades
st.subheader("Chemins des Tornades")
st.write("Cette section montre les chemins empruntés par les tornades.")

# Load data
def load_data(loc):
    return pd.read_csv(
        loc,
        parse_dates=['date'],
        dtype={'st':'category'},
    ).assign(
        elat=lambda d: d.elat.mask(d.elat == 0.0, d.slat),
        elon=lambda d: d.elon.mask(d.elon == 0.0, d.slon),
        mag=lambda d: d.mag.mask(d.mag == -9, 0),
    )

data_loc = 'data/Tornados 1950 - 2022.csv'
df_path = load_data(data_loc)

# Mapping magnitude to colors
mag_cm = {
    0: 'yellowgreen',
    1: 'gold',
    2: 'goldenrod',
    3: 'tomato',
    4: 'red',
    5: 'darkred',
}

# Extend folium.Map to add a method for adding tornado paths
def add_tornado_path_map(self, start_lat, start_long, end_lat, end_lon, color='blue', *args, **kwargs):
    folium.PolyLine(
        locations=[[start_lat, start_long], [end_lat, end_lon]],
        color=color,
        *args,
        **kwargs
    ).add_to(self)

folium.Map.add_tornado_path = add_tornado_path_map

# Extend pd.Series to add a method for adding tornado paths from DataFrame rows
def add_tornado_path_row(self, folium_map, slat='slat', slon='slon', elat='elat', elon='elon',
                         color_col='mag', cm=mag_cm, default_color='blue', *args, **kwargs):
    color = cm.get(self[color_col], default_color)
    folium_map.add_tornado_path(self[slat], self[slon], self[elat], self[elon], color, *args, **kwargs)

pd.Series.add_tornado_path = add_tornado_path_row

# Function to display tornado paths on a folium map using DataFrame
def show_tornado_paths(df, m=folium.Map(), *args, **kwargs):
    df.apply(lambda row: row.add_tornado_path(m, *args, **kwargs), axis=1)
    m.fit_bounds(m.get_bounds())
    return m

map_1950_1955 = show_tornado_paths(df_path.query('1950 <= yr <= 1955'))
st.markdown("### Chemins des Tornades (1950-1955)")
components.html(map_1950_1955._repr_html_(), height=600)

map_2016_2022 = show_tornado_paths(df_path.query('2016 <= yr <= 2022'))
st.markdown("### Chemins des Tornades (2016-2022)")
components.html(map_2016_2022._repr_html_(), height=600)

# Tornado Path Demonstration
data = df.copy()
data = data.sort_values('len', ascending=False)
data = data[['date', 'slat', 'slon', 'elat', 'elon', 'len']]
df_top_40 = data[0:40].reset_index(drop=True)

m = folium.Map(location=[42, -90], tiles='OpenStreetMap', zoom_start=5.0, min_zoom=2.0)
for i in range(40):
    folium.Marker(location=data.iloc[i, 1:3], icon=folium.Icon(color='green')).add_to(m)  # start point
    folium.Marker(location=data.iloc[i, 3:5], icon=folium.Icon(color='red')).add_to(m)  # end point
    points = [data.iloc[i, 1:3], data.iloc[i, 3:5]]
    line = PolyLine(locations=points, color="black", weight=3).add_to(m)
    folium.PolyLine(locations=[points[0], points[1]], color="red", weight=3).add_to(m)

st.markdown("### Chemins des 40 plus longues Tornades")
components.html(m._repr_html_(), height=600)

# Tornado outbreaks on '2011-04-27'
st.subheader("Événements Tornadiques du 27 Avril 2011 -- 207 TORNADES !!!")
st.write("Cette section montre les chemins empruntés par les tornades lors de l'événement tornadique du 27 avril 2011. [207 Tornado outbreaks on '2011-04-27'](https://www.washingtonpost.com/weather/2021/04/26/tornado-super-outbreak-april-2011/)")
df_2011 = data[data['date'] == '2011-04-27']
df_2011 = df_2011.reset_index(drop=True)
df_2011['index'] = df_2011.index.tolist()

m_2011 = folium.Map(location=[34, -88], tiles='OpenStreetMap', zoom_start=5.0, min_zoom=2.0)
for i in range(len(df_2011)):
    folium.Marker(location=df_2011.iloc[i, 1:3], icon=folium.Icon(color='green')).add_to(m_2011)  # start point
    folium.Marker(location=df_2011.iloc[i, 3:5], icon=folium.Icon(color='red')).add_to(m_2011)  # end point
    points = [df_2011.iloc[i, 1:3], df_2011.iloc[i, 3:5]]
    line = PolyLine(locations=points, color="black", weight=3).add_to(m_2011)
    folium.PolyLine(locations=[points[0], points[1]], color="red", weight=3).add_to(m_2011)

st.markdown("### Chemins des Tornades du 27 Avril 2011")
components.html(m_2011._repr_html_(), height=600)

# Conclusion
st.subheader("Conclusions et Résultats de l'EDA")
st.write("""
En résumé, l'EDA a fourni des insights précieux pour comprendre la distribution, la saisonnalité et l'impact des tornades aux États-Unis. Les analyses ont révélé plusieurs points clés :
- **Distribution Géographique** : Les tornades sont principalement concentrées dans le centre des États-Unis, avec une forte densité dans la Tornado Alley, particulièrement au Texas.
- **Saisonnalité** : Les tornades se produisent principalement au printemps et en été, avec un pic en mai et juin. La plupart des tornades surviennent l'après-midi et en début de soirée.
- **Tendances Temporelles** : Une tendance à la hausse du nombre de tornades observées annuellement a été notée, influencée par l'amélioration des technologies de détection.
- **Magnitude** : Les tornades de moindre intensité (magnitude 0 et 1) dominent les données, tandis que les tornades de forte intensité sont moins fréquentes mais causent des dommages plus importants.

Les résultats de l'EDA montrent que le Texas est l'État le plus touché par les tornades en termes de nombre total d'occurrences, de blessures, de décès et de pertes financières. En particulier :
- **Nombre de Tornades** : Le Texas enregistre le plus grand nombre de tornades parmi tous les états.
- **Impacts Humains** : Le Texas a le plus grand nombre de blessures et de décès causés par les tornades.
- **Pertes Financières** : Les pertes financières dues aux tornades sont les plus élevées au Texas.
""")
