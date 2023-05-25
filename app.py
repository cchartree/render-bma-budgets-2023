

import streamlit as st
#import plost
import PyPDF2 as ppdf
import pandas as pd
import numpy as np
import altair as alt


src = pd.DataFrame()
p01 = []

pdfobj = open('ข้อบัญญัติกรุงเทพฯ ว่าด้วยงบประมาณรายจ่ายประจำปี 2566.pdf', 'rb')
pdffile = ppdf.PdfReader(pdfobj)
npages = len(pdffile.pages[5:])

begpage = 5


def cleanpdf(pg):
    # Reading texts within pdf file
    df = pdffile.pages[pg].extract_text().strip()

    # Break items by the end of each numbers, followed by formatting
    df = df.replace('บาท\n', '|').replace('\n', ' ').replace('|', '\n').replace('บาท', '')
    df = df.replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ')
    df = df.replace(',', '')

    # After formatting, break text objects into lines, and feed into DataFrame
    df = df.splitlines()
    df = pd.DataFrame(df)

    # Take the last part to get only the numbers, and merge back to the DataFrame
    df = df.rename(columns={0: 'Raw'})
    df['Right'] = df['Raw'].str[-15:]
    df['Right2'] = df['Right'].str.split()
    df['Budgets'] = [n[-1] for n in df['Right2']]
    #     df['Budgets'] = df['Right'].str.extractall('(\d+)').unstack()
    #     df['Budgets'] = [s[e:] for (s, e) in zip(df['Right'], df['N'])]

    # Truncate only the texts (no numbers), as labels
    df['Raw2'] = [s.replace(r, '') for (s, r) in zip(df['Raw'], df['Budgets'])]
    #     df['Raw2'] = df['Raw'].replace(df['Budgets'],'')
    #     df['Raw2'] = [s[0:e] for (s, e) in zip(df['Raw'], df['L'])]

    # Rename columns
    df = df[['Raw2', 'Budgets']]

    # Assign flags for headers for hierarchies, as well as page numbers
    df['h1'] = df['Raw2'].str.contains(r'\(\d+\)').astype('int')
    df['h2'] = df['Raw2'].str.contains(r'^[กขค]\.').astype('int')
    df['flag page'] = df['Raw2'].str.contains(r'^\d+\ ').astype('int')
    df['T'] = np.where(df['flag page'] == 1, df['Raw2'].str.find(' '), 0)

    # Assign values based on flags
    df['Page'] = pd.to_numeric([s[0:e] for (s, e) in zip(df['Raw2'], df['T'])])
    df['Raw2'] = [s[e:] for (s, e) in zip(df['Raw2'], df['T'])]
    df['h1desc'] = np.where(df['h1'] == 1, df['Raw2'], np.nan)
    df['h2desc'] = np.where(df['h2'] == 1, df['Raw2'], np.nan)

    # Drop columns
    df = df.drop(columns={'T', 'flag page'})

    return df


p01 = cleanpdf(begpage)
print(p01['Raw2'][0])
p01


batch_from = 6
batch_to = 86

for i in np.arange(batch_from,batch_to):
    print(str("{:02}".format(i)), end=' ')
    ii = int(i)
    p02 = cleanpdf(ii)
    p01 = pd.concat([p01, p02])


len(p01['Page'].drop_duplicates())


p03 = p01
p03['Budgets'] = p03['Budgets'].str.extract('(\d+)').astype('int')
p03['Budgets Mn'] = (p03['Budgets'] / 1e6).astype('float64')
p03['Page'] = p03['Page'].fillna(method='ffill')
p03['h1desc'] = p03['h1desc'].fillna(method='ffill')
p03['h2desc'] = p03['h2desc'].fillna(method='ffill')
p03['h2desc'] = p03['h2desc'].fillna('')
p03.tail(10)


p04 = p03[(p03['h1'] == 0) & (p03['h2'] == 0)]

# Removing leading blank spaces
p04['Raw2'] = np.where(p04['Raw2'].str.find(' ') == 0, p04['Raw2'].str[1:], p04['Raw2'])
p04['Raw2'] = np.where(p04['Raw2'].str.find('.') == 1, '0' + p04['Raw2'], p04['Raw2'])
p04['h1desc'] = np.where(p04['h1desc'].str.find(' ') == 0, p04['h1desc'].str[1:], p04['h1desc'])
p04['h1desc'] = np.where(p04['h1desc'].str.find(')') == 2, p04['h1desc'].str.replace('(','(0'), p04['h1desc'])

p04



list_topics = [
      'Generic Administrations'
    , 'Maintenances'
    , 'Projects'
    , 'Communities'
    , 'Floods'
    , 'District Offices'
    , 'Human Resources'
    , 'Revenues'
    , 'Trainings'
    , 'Public Relations'
    , 'Welfares'
    , 'Waste Water Treatments'
    , 'Hygienic Foods'
    , 'Non-smoking Zones'
    , 'Cleanings'
    , 'Disease Controls'
    , 'Law Enforcements'
    , 'Green Zones'
    , 'Traffic Managements'
]

list_keywords = [
      r'งานบริหารท'
    , r'รุงรักษา|งานดูแล|ปรับปรุง'
    , r'โครงการ'
    , r'พัฒนาชุมชน'
    , r'งานระบาย|ญหาน'
    , r'กงานเขต'
    , r'บุคลากร|บุคคล'
    , r'บรายได'
    , r'อบรม|หลักสูตร'
    , r'ประชาสัมพ'
    , r'สวัสดิการ'
    , r'บัดน(.*)เสีย'
    , r'อาหารปลอดภ'
    , r'เขตปลอดบุหรี่'
    , r'ความสะอาด'
    , r'โรค'
    , r'บังคับใช(.*)กฎหมาย'
    , r'สวน(.*)สีเขียว'
    , r'จราจร'
]




p05 = pd.DataFrame()
p05 = p04

# Assign tags based on detected texts
def tags_keywords(lab,kw):
    p05[lab] = p05['Raw2'].str.contains(kw).astype('int')
    print(lab, kw, p05[p05[lab] == 1][lab].count())

for i in range(len(list_topics)):
    tags_keywords(list_topics[i], list_keywords[i])

p06 = pd.DataFrame()
p06 = p05
p06['Chained Tags'] = 'All Categories'

chained_tags = p05.columns.to_list()[8:]
print(len(chained_tags))

for i in range(len(chained_tags)):
    tcol = 'tmpcol' + str(i)
    p06[tcol] = np.where(p06[chained_tags[i]] == 1, chained_tags[i], '')
    p06['Chained Tags'] = p06['Chained Tags'] + '_' + p06[tcol]
    p06 = p06.drop(tcol, axis=1)

p06['Chained Tags'].drop_duplicates()




list_topics2 = list_topics
list_topics2.append('All Categories')
print(list_topics2)



print(p06.shape)



listrank = [
      'h1desc'
    , 'h2desc'
    , 'Raw2'
]

numcol = ['Budgets Mn'] * len(listrank)

tmp = p06

for g , n in zip(listrank, numcol):
    tmp2 = tmp[[g,n]].groupby([g], as_index=False)[n].agg('sum')
    tmp2 = tmp2.sort_values(by=[n], ascending=False)
    rcol = 'Budget Ranks ' + g
    tmp2[rcol] = tmp2[n].rank(method='dense', ascending=False).astype('int')
    p06 = pd.merge(p06, tmp2[[g,rcol]], how='left', on=[g])

p06.head()



disp = p06



# ------------------------------------------------------------------------------



st.set_page_config(layout='wide', initial_sidebar_state='expanded')

st.sidebar.header('Apply filters here to see only what you would like to see')

w_keywords = st.sidebar.multiselect('Categories by keywords (choose all that apply):',
                                    list_topics2, ['All Categories'])
w_minbudgets = st.sidebar.slider('Minimum Budget (THB millions):', min_value=0, max_value=5000,
                                 value=0, step=100)
w_maxbudgets = st.sidebar.slider('Maximum Budget (THB millions):', min_value=1000, max_value=20000,
                                 value=20000, step=100)
w_topranks = st.sidebar.slider('Topranks:', min_value=5, max_value=200,
                               value=20, step=1)
w_textsearch = st.sidebar.text_area('Category (in Thai) contains:', ' ')



numcol = 'Budgets Mn'

st.markdown('### BMA Budget Allocations for Fiscal Year 2023')
col1 = st.columns(1)
with col1:
    chains = '|'.join(w_keywords)
    dfplot = disp[
        (disp['Chained Tags'].str.contains(chains))
        & (disp['h1desc'].str.contains(w_textsearch))
        ]

    dfplot2 = dfplot[['h1desc', numcol]].groupby(['h1desc'], as_index=False)[numcol].agg('sum')
    dfplot2 = dfplot2.sort_values(by=[numcol]).tail(w_topranks)
    dfplot2 = dfplot2[
        (dfplot2[numcol] >= w_minbudgets)
        & (dfplot2[numcol] <= w_maxbudgets)
        ]

    chart = (
            alt.Chart(dfplot2)
            .mark_bar()
            .encode(
                x=alt.X("value", type=numcol, title=""),
                y=alt.Y("index", type="h1desc", title=""),
                #color=alt.Color("variable", type="nominal", title=""),
                order=alt.Order("variable", sort="descending"),
                )
            )
    st.altair_chart(chart, use_container_width=True)



