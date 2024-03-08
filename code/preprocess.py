import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

#v2clrspct - rigorous admin
#v2svdomaut - domestic policy free from interference
#v2svinlaut - foreign policy free
#v2x_rule - rule of law
#v2x_regime_amb - type of regime (with ambiguous), v2x_accountability
#v2x_civlib - civil liberties index
#v2x_corr - corruption index
#democracy indices: v2x_polyarchy, v2x_libdem, v2x_partipdem, v2x_delibdem, v2x_egaldem
#country_name, year, country_text_id



def load_data():
    '''We first load and merge the data obtained from multiple sources'''

    #Loading from downloaded csvs
    vdems_df = pd.read_csv('../data/V-Dem-CY-Core-v13.csv')
    vdems_df = vdems_df[['country_name', 'country_id','country_text_id', 'year', 'v2clrspct', 'v2x_rule', 'v2x_regime_amb', 'v2x_civlib', 'v2x_corr']]
    vdems_df = vdems_df.rename(columns={'country_name': 'Entity','country_text_id': 'Code', 'year': 'Year','v2clrspct': 'rigor_admin', 'v2svdomaut': 'domestic_autonomy', 'v2svinlaut': 'foreign_autonomy', 'v2x_rule': 'rule_of_law', 'v2x_regime_amb': 'regime', 'v2x_civlib': 'civil_liberties', 'v2x_corr': 'corruption' })

    territory_df = pd.read_csv('../data/percentage-of-territory-controlled-by-government.csv')[['Code','Year','terr_contr_vdem_owid']]
    colonized_df = pd.read_csv('../data/years-colonized.csv')[['Code','Year','Years the country has been colonized']]
    colonized_df = colonized_df.rename(columns={'Years the country has been colonized': 'years_colonized'})
    capacity_df = pd.read_csv('../data/state-capacity-index.csv')[['Code','Year','State capacity estimate']]
    capacity_df = capacity_df.rename(columns={'State capacity estimate': 'state_capacity'})
    taxes_df = pd.read_csv('../data/tax-revenues-as-a-share-of-gdp-unu-wider.csv')[['Code','Year','Taxes including social contributions (as a share of GDP)']]
    taxes_df = taxes_df.rename(columns={'Taxes including social contributions (as a share of GDP)': 'taxation'})
    taxes_df[['taxation']]= taxes_df[['taxation']].apply(pd.to_numeric)

    hdi_df = pd.read_csv('../data/human-development-index.csv')

    inf_capacity_df = pd.read_stata('../data/information_capacity.dta')[["VDemcode","year", "infcap_pca"]]
    inf_capacity_df = inf_capacity_df.rename(columns={'VDemcode': 'country_id', 'year': 'Year', 'infcap_pca': 'inf_capacity'})

    #Merging:
    #print(len(hdi_df))
    vdems_df = vdems_df.dropna()
    vdems_df = vdems_df.merge(inf_capacity_df, on=['country_id', 'Year'], how='left')
    final_df = hdi_df.merge(vdems_df, on=['Code', 'Year'], how='inner')

    final_df = final_df.drop('Entity_y', axis=1)
    final_df = final_df.rename(columns={'Entity_x': 'Entity'})
    final_df = final_df.merge(colonized_df, on=['Code', 'Year'], how='left')
    final_df = final_df.merge(capacity_df, on=['Code', 'Year'], how='left')
    final_df = final_df.merge(taxes_df, on=['Code', 'Year'], how='left')
    final_df = final_df.merge(territory_df, on=['Code', 'Year'], how='left')
    final_df['taxation'] = final_df[['taxation']].apply(lambda a: a/100)
    final_df = final_df.rename(columns={'terr_contr_vdem_owid': 'territory_control'})
    #print(final_df.head())
    #print(final_df.dtypes)
    final_df.to_csv('../data/combined.csv')
    return final_df

def data_split(init_df):
    #Normalizing and splitting data into training and testing sets

    init_df = init_df[['Human Development Index', 'rigor_admin','rule_of_law','regime','civil_liberties','corruption','years_colonized','state_capacity','taxation','territory_control']]
    
    init_df = init_df.dropna()
    #print(init_df.head())
    cols = init_df.columns
    #df_scaled = normalize(init_df) 

    #df_scaled = pd.DataFrame(df_scaled, columns=cols) 
    df_encoded = pd.get_dummies(init_df, columns=['regime'], drop_first=True) #one hot encoding regime types and admin scale
    #print(df_scaled.head())
    y=init_df['Human Development Index']
    X=init_df.drop(columns=['Human Development Index'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1680)
    return X_train, X_test, y_train, y_test



