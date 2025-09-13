import json
from bs4 import BeautifulSoup
from typing import Dict, List, Set, Tuple, Any
import os


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

        self._init_table()
        self._init_header()


    def _init_header(self):
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
            no_merge_cell = True
            for cid in range(self.col_header_boundary, self.max_cols):
                if self.type_table[rid][cid].startswith("merge_"):
                    no_merge_cell = False
                    break
            if no_merge_cell:
                if rid + 1 >= self.row_header_boundary:
                    self.row_header_boundary = rid + 1
                self.row_data_boundary = self.row_header_boundary
                break
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
            self.merge_counter += 1
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


def process_table_trees(tables, table_description):
    table_trees = []
    for table_id, table_html in enumerate(tables):
        table_tree = TableStructure(table_html, table_id, table_description)
        table_trees.append(table_tree)
    return table_trees
