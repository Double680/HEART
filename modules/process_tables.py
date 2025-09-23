import json
from bs4 import BeautifulSoup


class TableStructure:
    def __init__(self, table_html, table_id, table_description):
        self.table_html = table_html
        self.table_id = table_id
        self.table_description = table_description
        
        self.table_soup = BeautifulSoup(self.table_html, 'html.parser')

        self.merge_cells = 0
        self.max_rows = 0
        self.max_cols = 0
        self.row_data_boundary = 100000
        self.col_data_boundary = 100000
        self.row_header_boundary = 1
        self.col_header_boundary = 1
        self.occupied_cells = set()
        self.content_table = [['' for _ in range(100)] for _ in range(100)]
        self.type_table = [['' for _ in range(100)] for _ in range(100)]
        self.row_headers = {}
        self.col_headers = {}
        self.row_parents = {}

        self._init_table()
        self._init_header()
        self._get_table_headers()


    def _init_header(self):
        self.row_header_boundary = min(self.row_header_boundary, self.row_data_boundary)
        self.col_header_boundary = min(self.col_header_boundary, self.col_data_boundary)
        row_id = 0
        while True:
            if self.type_table[row_id][0] != 'empty' or row_id == self.row_data_boundary:
                break
            row_id += 1
            self.row_header_boundary = row_id
        col_id = 1
        while True:
            flag = True
            for row_id in range(self.row_header_boundary):
                if self.type_table[row_id][col_id] != 'empty':
                    flag = False
                    break
            if not flag or self.col_header_boundary >= self.col_data_boundary:
                break
            col_id += 1
            self.col_header_boundary = col_id

        for rid in range(self.row_data_boundary):
            merge_cells = 0
            for cid in range(self.col_header_boundary, self.max_cols):
                cell_type = self.type_table[rid][cid]
                if cell_type.startswith("merge_"):
                    rowspan = int(cell_type.split('_')[3])
                    merge_cells = max(merge_cells, rowspan)
            merge_cells -= 1
            if merge_cells == -1:
                if rid + 1 >= self.row_header_boundary:
                    self.row_header_boundary = rid + 1
                self.row_data_boundary = self.row_header_boundary
                break
        self.row_header_boundary = self.row_data_boundary
        self.col_data_boundary = self.col_header_boundary


    def _init_table(self):
        rows = self.table_soup.find_all('tr')
        max_rows = len(rows)
        max_cols = 0
        for row_id, row in enumerate(rows):
            col_id = 0
            cells = row.find_all(['td', 'th'])
            for cell in cells:
                while (row_id, col_id) in self.occupied_cells:
                    col_id += 1
                rowspan = int(cell.get('rowspan', 1))
                colspan = int(cell.get('colspan', 1))
                self._fill_cell(cell, row_id, col_id, rowspan, colspan)
                col_id += colspan
            max_cols = max(max_cols, col_id)
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.row_data_boundary = min(self.row_data_boundary, self.max_rows)
        self.col_data_boundary = min(self.col_data_boundary, self.max_cols)
        


    def _get_cell(self, cell, row_id, col_id, rowspan, colspan):
        
        table_id = self.table_id
        cell_text = cell.get_text(strip=True)

        if not cell_text or cell_text.strip() in ['']:
            cell_type = 'empty'
        elif cell_text.strip() in ['-', '—'] or f"{table_id}-{row_id}-{col_id}" in self.table_description:
            cell_type = 'data'
        elif rowspan > 1 or colspan > 1:
            self.merge_cells += 1
            cell_type = f'merge_{row_id}_{col_id}_{rowspan}_{colspan}'
        else:
            cell_type = ''
            
        return cell_text, cell_type


    def _fill_cell(self, cell, row_id, col_id, rowspan, colspan):
        cell_text, cell_type = self._get_cell(cell, row_id, col_id, rowspan, colspan)
        if cell_type == "data":
            self.row_data_boundary = min(self.row_data_boundary, row_id)
            self.col_data_boundary = min(self.col_data_boundary, col_id)
        for i in range(row_id, row_id + rowspan):
            for j in range(col_id, col_id + colspan):
                self.content_table[i][j] = cell_text
                self.type_table[i][j] = cell_type
                self.occupied_cells.add((i, j))


    def _get_table_headers(self):
        row_header_stack = []
        for row_id in range(self.row_header_boundary, self.max_rows):
            row_name = []
            for col_id in range(self.col_header_boundary):
                row_name.append(self.content_table[row_id][col_id])
            self.row_headers[row_id] = row_name
    
            parent_flag = True
            for col_id in range(self.col_header_boundary, self.max_cols):
                if self.type_table[row_id][col_id] != "empty":
                    parent_flag = False
                    break
            
            if parent_flag:
                if row_id > 0 and row_id - 1 in row_header_stack:
                    row_header_stack.append(row_id)
                else:
                    row_header_stack = []
                    row_header_stack.append(row_id)
            if len(row_header_stack):
                self.row_parents[row_id] = row_header_stack[-1]
            else:
                self.row_parents[row_id] = row_id
        for row_id in range(self.row_header_boundary, self.max_rows):
            if self.row_parents[row_id] != row_id:
                self.row_headers[row_id] = self.row_headers[self.row_parents[row_id]] + self.row_headers[row_id]

        for col_id in range(self.col_header_boundary, self.max_cols):
            col_name = []
            for row_id in range(self.row_header_boundary):
                col_name.append(self.content_table[row_id][col_id])
            self.col_headers[col_id] = col_name


    def extract_subtable(self, row_ids, col_ids):
        row_ids = sorted(row_ids)
        col_ids = sorted(col_ids)

        subtable_content = [[
            self.content_table[i][j] for j in col_ids
        ] for i in row_ids]
        subtable_type = [[
            self.type_table[i][j] for j in col_ids
        ] for i in row_ids]

        row_num = len(row_ids)
        col_num = len(col_ids)
        subtable_soup = BeautifulSoup('', 'html.parser')
        
        table = subtable_soup.new_tag('table')
        processed_cells = set()

        for i in range(row_num):
            tr = subtable_soup.new_tag('tr')

            j = 0
            while j < col_num:
                if (i, j) in processed_cells:
                    j += 1
                    continue
                content = subtable_content[i][j]
                cell_type = subtable_type[i][j]
                
                td = subtable_soup.new_tag('td')
                td.string = content

                if cell_type.startswith('merge_'):
                    rowspan = 1
                    while i + rowspan < row_num and subtable_type[i + rowspan][j] == cell_type:
                        rowspan += 1
                    colspan = 1
                    while j + colspan < col_num and subtable_type[i][j + colspan] == cell_type:
                        colspan += 1

                    for r in range(i, i+rowspan):
                        for c in range(j, j+colspan):
                            processed_cells.add((r, c))
                    if rowspan > 1:
                        td['rowspan'] = str(rowspan)
                    if colspan > 1:
                        td['colspan'] = str(colspan)
                else:
                    processed_cells.add((i, j))         

                tr.append(td)
                j += 1

            table.append(tr)   

        subtable_html = str(table)
        return subtable_html
    

    def list_row_headers(self):
        content = ""
        for key in self.row_headers:
            header = self.row_headers[key]
            cleaned_header = []
            for item in header:
                if len(item.strip()):
                    cleaned_header.append(item)
            content = content + '- ' + str(key) + ": " + ' -> '.join(cleaned_header) + '\n'
        return content


    def list_col_headers(self):
        content = ""
        for key in self.col_headers:
            header = self.col_headers[key]
            cleaned_header = []
            for item in header:
                if len(item.strip()):
                    cleaned_header.append(item)
            content = content + '- ' + str(key) + ": " + ' -> '.join(cleaned_header) + '\n'
        return content


def process_table_trees(tables, table_description):
    table_trees = []
    for table_id, table_html in enumerate(tables):
        table_tree = TableStructure(table_html, table_id, table_description)
        table_trees.append(table_tree)
    return table_trees
