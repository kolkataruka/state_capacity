from turtle import pd
import pandas as pd

#v2clrspct - rigorous admin
#v2svdomaut - domestic policy free from interference
#v2svinlaut - foreign policy free
#v2x_rule - rule of law
#v2x_regime_amb - type of regime (with ambiguous), v2x_accountability
#v2x_civlib - civil liberties index
#v2x_corr - corruption index
#democracy indices: v2x_polyarchy, v2x_libdem, v2x_partipdem, v2x_delibdem, v2x_egaldem
#country_name, year, country_text_id

vdems_df = pd.read_csv('../data/V-Dem-CY-Core-v13.csv')
vdems_df = vdems_df[['country_name', 'country_text_id', 'year', 'v2clrspct', 'v2svdomaut', 'v2svinlaut', 'v2x_rule', 'v2x_regime_amb', 'v2x_civlib', 'v2x_corr']]
vdems_df = vdems_df.rename(columns={'v2clrspct': 'rigor_admin', 'v2svdomaut': 'domestic_autonomy', 'v2svinlaut': 'foreign_autonomy', 'v2x_rule': 'rule_of_law', 'v2x_regime_amb': 'regime', 'v2x_civlib': 'civil_liberties', 'v2x_corr': 'corruption' })

territory_df = pd.read_csv('../data/percentage-of-territory-controlled-by-government.csv')[['Entity','Code','Year','terr_contr_vdem_owid']]
colonized_df = pd.read_csv('../data/years-colonized.csv')[['Entity','Code','Year','Years the country has been colonized']]
colonized_df = colonized_df.rename(columns={'Years the country has been colonized': 'years_colonized'})
capacity_df = pd.read_csv('../data/state-capacity-index.csv')
taxes_df = pd.read_csv('../data/tax-revenues-as-a-share-of-gdp-unu-wider.csv')
hdi_df = pd.read_csv('../data/human-development-index.csv')
