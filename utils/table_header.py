def get_table_docs(sample):
    table_docs = []

    def col_dfs(item, text):
        col_site = item['col_coords']
        col_text = text
        if item['text'] != '':
            col_text = col_text + ' - ' + item['text']
        if len(item['children']) == 0:
            col_docs.append({
                'site': col_site,
                'text': col_text,
            })
            return
        for child in item['children']:
            col_dfs(child, col_text)

    def row_dfs(item, text):
        row_site = item['row_coords']
        row_text = text
        if item['text'] != '':
            row_text = row_text + ' - ' + item['text']
        if len(item['children']) == 0:
            row_docs.append({
                'site': row_site,
                'text': row_text,
            })
            return
        for child in item['children']:
            row_dfs(child, row_text)

    for table in sample:
        col_headers = table['col_headers']
        row_headers = table['row_headers']
        col_docs, row_docs = [], []
        for col_header in col_headers:
            text = 'COLUMN'
            col_dfs(col_header, text)
        for row_header in row_headers:
            text = 'ROW'
            row_dfs(row_header, text)

        table_docs.append({
            "col_docs": col_docs,
            "row_docs": row_docs
        })
    return table_docs

def get_site_lists(table_docs):
    col_sites, row_sites = [], []
    for i, table in enumerate(table_docs):
        col_sites += [(i, 'col', table['col_docs'][id]['site']) for id in range(len(table['col_docs']))]
        row_sites += [(i, 'row', table['row_docs'][id]['site']) for id in range(len(table['row_docs']))]

    return col_sites, row_sites