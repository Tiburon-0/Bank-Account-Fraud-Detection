import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, dash_table, Input, Output

# ── Color constants ────────────────────────────────────────────────────────────
# Defined once here so callbacks stay DRY and colors match the CSS design tokens.
# Changing a color here updates every chart automatically.
color_fraud = '#f87171'   # matches --alert-crimson in style.css
color_legit = '#34d399'   # matches --neon-green in style.css
color_map = {'Fraudulent': color_fraud, 'Legitimate': color_legit}

# ── Feature registry ───────────────────────────────────────────────────────────
features = {
    # target — kept in dict so callbacks can reference it, excluded from dropdowns below
    'fraud_bool':                       ['Fraud Label',                         '1 = Fraudulent Application, 0 = Legitimate'],

    # demographics
    'income':                           ['Income',                              'Annual Income of the Applicant | Normalized 0–1'],
    'customer_age':                     ['Customer Age',                        'Age in Decade Bins (20 = Twenties, 30 = Thirties, etc.)'],
    'employment_status':                ['Employment Status',                   'Employment Status | 7 Anonymized Values (CA–CG)'],
    'housing_status':                   ['Housing Status',                      'Residential Status | 7 Anonymized Values (BA–BF)'],

    # application metadata
    'payment_type':                     ['Payment Type',                        'Credit Payment Plan Type | 5 Anonymized Values (AA–AE)'],
    'proposed_credit_limit':            ['Proposed Credit Limit',               "Applicant's Requested Credit Limit | Ranges [200, 2000]"],
    'intended_balcon_amount':           ['Intended Balance',                    'Transfer Amount at Application | Negative Values are Suspicious'],
    'days_since_request':               ['Days Since Request',                  'Days Between Application Submission and Processing | Ranges [0, 78]'],
    'foreign_request':                  ['Foreign Request',                     'Application Originated from a Foreign IP Address'],
    'source':                           ['Application Source',                  'Channel: Browser (INTERNET) or Mobile App (APP)'],
    'month':                            ['Month',                               'Month the Application was Made | Ranges [0, 7]'],

    # identity/contact signals
    'name_email_similarity':            ['Name–Email Similarity',               "Similarity Between Email and Applicant's Name | High Similarity = Lower Risk | Ranges [0, 1]"],
    'email_is_free':                    ['Free Email Provider',                 'Email from Free Provider (Gmail, Yahoo) — Free Emails are Higher Risk'],
    'phone_home_valid':                 ['Home Phone Valid',                    'Home Phone Number Verified'],
    'phone_mobile_valid':               ['Mobile Phone Valid',                  'Mobile Phone Number Verified'],
    'prev_address_months_count':        ['Prev. Address Duration',              'Months at Previous Address | −1 = Not Provided | Ranges [−1, 380]'],
    'current_address_months_count':     ['Current Address Duration',            'Months at Current Address | −1 = Missing | Ranges [−1, 406]'],
    'bank_months_count':                ['Bank Account Age',                    'Age of Previous Bank Account in Months | −1 = Not Held | Ranges [−1, 31]'],
    'has_other_cards':                  ['Has Other Cards',                     'Applicant Holds Other Cards with This Bank'],

    # velocity — strongest fraud signals
    'velocity_6h':                      ['Velocity (6h)',                       'Avg Applications per Hour in the Last 6 Hours | Ranges [−211, 24763]'],
    'velocity_24h':                     ['Velocity (24h)',                      'Avg Applications per Hour in the Last 24 Hours | Ranges [1329, 9527]'],
    'velocity_4w':                      ['Velocity (4 Weeks)',                  'Avg Applications per Hour in the Last 4 Weeks | Ranges [2779, 7043]'],
    'zip_count_4w':                     ['ZIP Count (4w)',                      'Applications from Same ZIP Code in Last 4 Weeks | Ranges [1, 5767]'],
    'bank_branch_count':                ['Bank Branch Count',                   'Applications at Same Branch in Last 8 Weeks | Ranges [0, 2521]'],
    'date_of_birth_distinct_emails_4w': ['DOB Email Count (4w)',                'Distinct Emails for Same Date of Birth in Last 4 Weeks | High = Identity Farming | Ranges [0, 42]'],

    # device/session
    'device_os':                        ['Device OS',                           'Operating System of Requesting Device | Windows, macOS, Linux, X11, or Other'],
    'device_distinct_emails_8w':        ['Device Emails for Past 8 Weeks',      'Distinct Emails from This Device in Last 8 Weeks | Ranges [0, 3]'],
    'device_fraud_count':               ['Device Fraud Count',                  'Fraudulent Applications Submitted from This Device | Ranges [0, 1]'],
    'session_length_in_minutes':        ['Session Length in Minutes',           'Length of User Session in Minutes | −1 = Unknown | Ranges [−1, 107]'],
    'keep_alive_session':               ['Keep-Alive Session',                  'Session Was Kept Alive — Indicative of Bot Behavior'],
    'credit_risk_score':                ['Credit Risk Score',                   'Bureau Credit Risk Score | Ranges [−176, 387]'],
}


# dictionary of lists for direct access and structured iteration

# radio button group
risk_profile_groups = {
    'Velocity': ['velocity_6h', 'velocity_24h', 'velocity_4w', 'zip_count_4w'],
    
    'Device_Session': ['foreign_request', 'source', 'email_is_free', 'device_os', 'device_distinct_emails_8w',
                        'device_fraud_count', 'session_length_in_minutes', 'keep_alive_session'],
    
    'Identity_Contact': ['income', 'customer_age', 'proposed_credit_limit', 'intended_balcon_amount', 'name_email_similarity',
                          'phone_home_valid', 'phone_mobile_valid', 'prev_address_months_count', 'current_address_months_count',
                           'bank_months_count', 'has_other_cards', 'date_of_birth_distinct_emails_4w', 'credit_risk_score']
}

feature_groups = {f: risk_profile_groups[f] for f in ('Velocity', 'Device_Session')}

graph_ids = {
    'histogram'   : 'device_histogram',
    'box_plot'    : 'device_box',
    'violin_plot' : 'risk_group_violin',
    'scatter_plot' : 'risk-scatter',
}

graph_input_ids = {
    'section_1' : {'input_1' : 'device_feature', 'input_2' : 'fraud_filter'},
    'section_2' : {'input_1' : 'risk_group'},
    'section_3' : {'input_x' : 'x-selection', 'input_y' : 'y-selection', 'slider_input' : 'sample-slider'}
}


# ==========================[App Initialization]==========================

# instantiate app and save server attribute
app = Dash(__name__)
server = app.server

# load data
df = pd.read_csv('Base.csv')

# maps 'fraud_bool' values to an additional column titled 'Class' fraudulent vs. legit accounts
df['Class'] = df['fraud_bool'].map({1: 'Fraudulent', 0: 'Legitimate'})

# KPI values
total_n = len(df)
fraud_n = int(df['fraud_bool'].sum())
legit_n = total_n - fraud_n
fraud_rate = fraud_n / total_n


# ===============================[Layout]===============================
app.layout = html.Div([

    # ==============[Title Section]==============

    # header and paragraph
    html.Div([  # title container 
        html.H1('Bank Account Fraud: A Signal Analysis'),  # header
        html.H2('Interactive Fraud Signal Explorer | NeurIPS 2022 Dataset | 1,000,000 Applications'),

        dcc.Markdown('''Dating back to the dawn of the modern internet in the 1980s, humanity's interconnectivity has grown in parallel with its technological dependence. As information becomes increasingly
accessible and expansive, societies across the globe shift away from tangible tender toward digital banking environments. While architected to optimize security, these environments
and their constituents fall victim to human adaptation and exploitation. Human nature reveals that where there lies a will, there stands to be a way. Thus, financial
fraud continues to be a primary avenue through which criminals and organizations amass wealth while subverting legal restrictions. In fact, the United Nations Office on Drugs and Crime
 asserts that 2.7% of global gross domestic product (GDP) falls within the umbrella of financial fraudulence, specifically money laundering [Cheng et al., 2023.](https://ieeexplore-ieee-org.du.idm.oclc.org/document/10114503)''', 
 link_target='_blank'),

        dcc.Markdown('''In connection, bank account fraud remains a popular method through which criminals and their cabals perpetrate money laundering. A host of these organizations are
        notorious for thriving off pipelines of terrorist financing, narcotics trafficking, and human trafficking. This necessitates not only the inauguration of committees such
        as the Financial Action Task Force (FATF), but also the implementation of more sophisticated fraud-detection pipelines, i.e., Artificial Intelligence (AI) and
        machine-learning (ML) classification models. The analyses provided below, constructed using anonymized Bank Account application data published during the 2022
        NeurIPS conference, enable dissection of real-world fraud signal indicators and their interactions in comparison with those of legitimate accounts. These signals span
        digital footprints, demographical data, temporal metrics, and of course, financial histories [Ofoeda et al., 2020.](https://onlinelibrary-wiley-com.du.idm.oclc.org/doi/full/10.1002/ijfe.2360)''',
        link_target='_blank')], 

        className='dark-header'),

    # ==============[Intro Section]==============
    # kpi badges
    html.Div([  # kpi section container

        html.Div([  # fraudulent count container
            html.P('Fraudulent Applications'),
            html.H2(f'{fraud_n:,}')],  # aggregation of fraudulent accounts
            className='kpi-badge fraud'),

        html.Div([  # legitimate count container
            html.P('Legitimate Applications'),
            html.H2(f'{legit_n:,}')],  # aggregation of legitimate accounts
            className='kpi-badge legit'),

        html.Div([ # fraud rate container
            html.P('Fraud Rate'),
            html.H2(f'{fraud_rate:.2%}')], # fraud rate
            className='kpi-badge'),

    ], className='layout-split'),

    # ==============[Descriptive Statistics Section]==============

    # datatable for descriptive statistics
    html.Div([  # container for datatable

        html.H2(children='Account Dashboard'),

        # segue into datatable for class imbalance modeling
        dcc.Markdown('''This dataset models the real-world class imbalance of fraudulent and legitimate accounts.
                      Notice the low frequency of the fraudulent accounts against their legitimate counterparts. With respect
                      to its vastness, only the initial 1000 accounts of this dataset are displayed in the datatable below. To retrieve
                      the full dataset, visit [Kaggle.](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022)''',
                      link_target='_blank'),

        # condensed display of dataset
        # 1000 rows shows rarity of fraudulence while preventing browser crashing from loading 1 million rows
        # 50 rows per page keeps readability without overwhelming user with excessive scrolling

        dash_table.DataTable(
            columns=[{'name': i, 'id': i} for i in df.columns],
            data=df.head(1000).to_dict('records'),
            page_size=50,
            style_table={'overflowX': 'auto'},
            style_cell={
                'minWidth': '120px',
                'maxWidth': '200px',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis'})
    ], className='dark-stats'),

    # ==============Section 1: Device Session Signal Indicator Analysis [histogram, boxplot]==============

    html.Div([  # section 1 container

        html.Div([
            html.H2(children='Device Signal Explorer') # section 1 header
        ]),

        # segue into section 1: device session signal indicator analysis
        dcc.Markdown('''When investigating fraud, authorities typically evaluate several indicators linked to applications.
                      One major indicator is the digital footprint of the suspected occurrence, which can be composed of several attributes,
                      some of which include the following column features of this dataset: `'foreign_request'`, `'email_is_free'`, `'device_os'`,
                      `'device_distinct_emails_8w'`, `'device_fraud_count'`, `'session_length_in_minutes'`, and `'keep_alive_session'`. The histogram
                      and boxplot below facilitate analysis of these features, specifically their distributions across the dataset. Users may explore the 
                     distribution of the aforementioned features with the option to filter the dataset by class via the provided dropdown and radio buttons
                      above the plots.'''),

        # prioritizing *device session* analysis over *velocity*
        # this enables analysis of an important feature while keeping dropdowns clean and manageable
        # i.e., not overwhelming the user with dropdown options 

        html.Div([ # container for device session analysis

            html.Div([  # container for dropdown
                
                html.Label('Device Session Features'),

                dcc.Dropdown(
                    options=[{'label': f, 'value': f}
                             for f in feature_groups['Device_Session']],
                    
                    value='foreign_request', # 'foreign_request' immediately raises eyebrows for authorities/administrators
                    id=graph_input_ids['section_1']['input_1'],
                    clearable=False,
                ),
            ], style={'flex': '1'}),

            html.Div([  # container for radiobuttons
                
                html.Label('Application Class'),

                dcc.RadioItems(
                    options=[
                        {'label': 'All Applications', 'value': 'total'},
                        {'label': 'Fraud Only', 'value': '1'},
                        {'label': 'Legitimate Only', 'value': '0'},
                    ],
                    value='total', # default value, but users can narrow if they want
                    id=graph_input_ids['section_1']['input_2'],
                    inputStyle={'marginRight': '6px'},
                ),
            ], style={'flex': '1'}),

        ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}),

        html.Div([
            html.Div([
                dcc.Graph(id=graph_ids['histogram']) # histogram id
            ]),
            html.Div([
                dcc.Graph(id=graph_ids['box_plot']) # boxplot id
            ]),
        ], className='layout-split'),

    ], className='dark-card'),

    # ==========Section 2: Risk Profile Analysis [violin plots]===========

    html.Div([  # container for section 2

        html.Div([
            html.H2(children='Risk Profiles')
        ]),

        # segue into section 2
        dcc.Markdown('''Tellingly, a dataset spanning 1,000,000 applicants can be quite diverse; 
                     column feature values vary by application, placing each application into one of
                      two classifications. Additionally, some applications (i.e., the fraudulent applications)
                      share the same origin. Certain features serve as stronger indicators when composing the
                      applicants' risk profiles. Thus, the violin plots below convey this diversity through analysis
                      of this dataset's categorical distribution. Each risk group and its associated radiobutton, `Velocity`,
                      `Device_Session`, and `Identity_Contact`, encompasses various subsects of features, with a devoted violin
                      plot for each feature.'''),

        html.Div([
            html.Label('Risk Group'),

            dcc.RadioItems(
                options=[{'label': g, 'value': g}
                         for g in risk_profile_groups],
                
                value='Velocity', # 'velocity' group features are strong fraud signals (i.e., indicators of abnormal behavior)
                id=graph_input_ids['section_2']['input_1'],
                inline=True,
                inputStyle={'marginRight': '6px'},
                labelStyle={'marginRight': '28px'},
            ),
        ], style={'marginBottom': '20px'}),

        dcc.Graph(id=graph_ids['violin_plot']), # violin plot graph id

    ], className='dark-card'),

    # ========section 3: bivariate risk analysis [scatter plot]========

    html.Div([  # section 3 container
        html.Div([

            html.Div([
                html.H2(children='Bivariate Risk Analysis')
            ]),

            # segue into section 3: bivariate risk analysis
            dcc.Markdown('''Analysis of a single feature by itself cannot accurately distinguish a fraudulent case from a legitimate case.
                         However, by assessing the dynamic relationships between multiple features, investigators and scholars alike can gain
                          insight into each column feature's partial dependence, (i.e., each signal indicator's marginal effect on the machine-learning
                          model's ability to classify an application as either `fraudulent` or `legitimate`). Users may explore the partial
                          dependence of each indicator, selecting for specific `x-variables` and `y-variables`, as well as `sample sizes`, via the dropdowns
                          and sample slider above the scatter plot.'''),

            html.Div([  # Risk Matrix Panel
                html.Div([
                    html.Label('X-Axis Feature'),
                    dcc.Dropdown(
                        options=[{'label': v[0], 'value': f}
                                 for f, v in features.items() if f != 'fraud_bool'], # exclusion of target variable fraud_bool
                        
                        value='foreign_request',
                        id=graph_input_ids['section_3']['input_x'],
                        clearable=False,
                    ),
                ], style={'flex': '1'}),

                html.Div([
                    html.Label('Y-Axis Feature'),
                    dcc.Dropdown(
                        options=[{'label': v[0], 'value': f}
                                 for f, v in features.items() if f != 'fraud_bool'],

                        value='velocity_6h', # high velocity raises eyebrows when paired with recency
                        id=graph_input_ids['section_3']['input_y'],
                        clearable=False,
                    ),
                ], style={'flex': '1'}),

                html.Div([
                    html.Label('Sample Size'),

                    dcc.Slider(
                        min=1000, max=50000, step=None, # max of 50000 shows scale without overwhelming user
                        marks={1000: {'label': '1k', 'style': {'color': '#9ca3af', 'font-size':'11px'}},
                               5000: {'label': '5k', 'style': {'color': '#9ca3af', 'font-size':'11px'}}, 
                               10000: {'label': '10k', 'style': {'color': '#9ca3af', 'font-size':'11px'}},
                               25000: {'label': '25k', 'style': {'color': '#9ca3af', 'font-size':'11px'}},
                               50000: {'label': '50k', 'style': {'color': '#9ca3af', 'font-size':'11px'}},
                               },
                        value=10000,
                        id=graph_input_ids['section_3']['slider_input'],
                    ),
                ], style={'flex': '2', 'paddingTop': '8px'}),
            ], style={'display': 'flex', 'gap': '20px', 'alignItems': 'flex-start',
                      'marginBottom': '20px'}),

            dcc.Graph(id=graph_ids['scatter_plot']) # scatter plot graph id

        ], className='dark-card'),

    ]),
], className='dark-container')


# Callbacks

# =========[Callback for Section 1: Histogram]==========

@app.callback(
    Output(graph_ids['histogram'], 'figure'),  # histogram figure [left]
    Output(graph_ids['box_plot'],  'figure'),  # box plot figure [right]
    Input(graph_input_ids['section_1']['input_1'],    'value'), # controls device_feature dropdown
    Input(graph_input_ids['section_1']['input_2'],    'value'), # controls radiobuttons [total, fraud : 1, legitimate : 0]
)

def update_device(feature, fraud_filter):
    pio.templates.default = 'plotly_dark'

    temp = df.copy()  # prevents mutation of original dataframe

    if fraud_filter != 'total':
        temp = temp[temp['fraud_bool'] == int(fraud_filter)] # filters based on whether user selects 1 (fraudulent) or 0 (legitimate)

    label = features[feature][0]  # short name for chart title

    # Histogram
    fig_hist = px.histogram(
        temp,
        x=feature,
        color='Class',
        color_discrete_map=color_map,
        barmode='overlay',
        opacity=0.65,
        nbins=60,
        title=f'Distribution: {label}',
        labels={feature: label, 'count': 'Applications'},
        height=450
    )
    fig_hist.update_layout(legend_title_text='Class', bargap=0.05)

    # Box plot
    fig_box = px.box(
        df,                 # presents full, unfiltered dataframe
        x='Class',
        y=feature,
        color='Class',
        color_discrete_map=color_map,
        title=f'Fraud vs. Legitimate: {label}',
        labels={feature: label},
        height=450,
    )

    fig_box.update_traces(boxpoints=False) # saves computation expenditure to prevent browser from being overworked (i.e., computing 1M datapoints)
    fig_box.update_layout(showlegend=True)

    return fig_hist, fig_box

# =========[Callback for Section 2: Violin Plot]==========


@app.callback(
    Output(graph_ids['violin_plot'], 'figure'),  # violin plot figure
    Input(graph_input_ids['section_2']['input_1'], 'value')  # violin plot inputs
)

def update_group(group):
    pio.templates.default = 'plotly_dark'

    cols = risk_profile_groups[group]

    temp = df[cols + ['Class']].sample(n=50_000, random_state=415)

    temp_melted = temp.melt( # converts df copy from wide form to long form
        id_vars='Class', var_name='Feature', value_name='Value')

    temp_melted['Feature'] = temp_melted['Feature'].map( # maps features with their short labels for temp df
        lambda c: features[c][0])

    n_rows = -(-len(cols) // 3) # ceiling height

    fig = px.violin(
        temp_melted,
        x='Class',
        y='Value',
        color='Class',
        color_discrete_map=color_map,
        facet_col='Feature',
        facet_col_wrap=3,
        box=True,
        height=max(550, n_rows * 220),
        title=f'{group} Features - Distribution by Fraud Class',
    )

    fig.update_yaxes(matches=None, showticklabels=True)

    # handling plotly's auto-generation of feature titles for facet cols given to multiple violin plots
    # (i.e., 'Feature=Velocity (6h)')

    fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1])) 

    fig.update_layout(showlegend=True)

    return fig


# ===========[Callback for Section 3: Bivariate Risk Scatter Plot]

@app.callback(
    Output(graph_ids['scatter_plot'], 'figure'), # scatter plot figure
    Input(graph_input_ids['section_3']['input_x'], 'value'), # controls x-value dropdown
    Input(graph_input_ids['section_3']['input_y'], 'value'), # controls y-value dropdown
    Input(graph_input_ids['section_3']['slider_input'], 'value'), # controls sample_slider
)
def update_scatter(x_feat, y_feat, n_samples):
    pio.templates.default = 'plotly_dark'

    temp = (df
            .sample(n=min(n_samples, len(df)), random_state=415) # guard in case samples selected are greater than dataframe length
            .sort_values('fraud_bool'))

    fig = px.scatter(
        temp,
        x=x_feat, y=y_feat,
        color='Class',
        color_discrete_map=color_map,
        opacity=0.55,
        labels={
            x_feat: features[x_feat][0],
            y_feat: features[y_feat][0],
        },
        title=f'{features[x_feat][0]} vs. {features[y_feat][0]} (n={n_samples:,})',
        height=600,
    )

    fig.update_traces(marker=dict(size=3))
    fig.update_layout(legend_title_text='Class')

    return fig

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
