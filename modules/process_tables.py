import json
from bs4 import BeautifulSoup
from typing import Dict, List, Set, Tuple, Any
import os


class TableTree:
    def __init__(self, table_html, table_id, table_description):
        self.table_html = table_html
        self.table_id = table_id
        self.table_description = table_description
        
        self.table_soup = BeautifulSoup(self.table_html, 'html.parser')

        self.merge_cells = 0
        self.max_rows = 0
        self.max_cols = 0
        self.row_boundary = 0
        self.col_boundary = 0
        self.occupied_cells = set()

        self._init_table()
        self._fill_matrices()


    def _init_table(self):
        rows = self.table_soup.find_all('tr')
        max_rows = len(rows)
        max_cols = 0
        for row in rows:
            current_col = 0
            for cell in row.find_all(['td', 'th']):
                colspan = int(cell.get('colspan', 1))
                current_col += colspan
            max_cols = max(max_cols, current_col)
        self.max_rows = self.row_boundary = max_rows
        self.max_cols = self.col_boundary = max_cols
        self.content_matrix = [['' for _ in range(self.max_cols)] for _ in range(self.max_rows)]
        self.type_matrix = [['' for _ in range(self.max_cols)] for _ in range(self.max_rows)]


    def _fill_matrices(self):
        rows = self.table_soup.find_all('tr')
        for row_id, row in enumerate(rows):
            col_id = 0
            cells = row.find_all(['td', 'th'])
            for cell in cells:
                while (row_id, col_id) in self.occupied_cells and col_id < self.max_cols:
                    col_id += 1
                if col_id >= self.max_cols:
                    break
                
                rowspan = int(cell.get('rowspan', 1))
                colspan = int(cell.get('colspan', 1))
                
                cell_text, cell_type = self._get_cell(cell, row_id, col_id)
                if cell_type == 'data':
                    self.row_boundary = min(self.row_boundary, row_id)
                    self.col_boundary = min(self.col_boundary, col_id)

                self._fill_cell(row_id, col_id, rowspan, colspan, cell_text, cell_type)

                col_id += colspan

        if self.row_boundary == self.max_rows:
            self.row_boundary = 0
        if self.col_boundary == self.max_cols:
            self.col_boundary = 0

        if self.row_boundary > 0:
            flag = True
            for col in range(self.col_boundary, self.max_cols):
                if self.type_matrix[self.row_boundary - 1][col] != 'empty' and self.content_matrix[self.row_boundary - 1][col].strip():
                    flag = False
                    break
        if flag and self.content_matrix[self.row_boundary-1][0].strip():
            self.row_boundary -= 1


    def _get_cell(self, cell, row_id, col_id):
        
        table_id = self.table_id
        cell_text = cell.get_text(strip=True)

        if not cell_text or cell_text.strip() in ['']:
            cell_type = 'empty'
        elif cell_text.strip() in ['-', '—'] or f"{table_id}-{row_id}-{col_id}" in self.table_description:
            cell_type = 'data'
        elif int(cell.get('colspan', 1)) > 1 or int(cell.get('rowspan', 1)) > 1:
            self.merge_counter += 1
            cell_type = f'merge_{self.merge_counter}'
        else:
            cell_type = ''
            
        return cell_text, cell_type


    def _fill_cell(self, row_id, col_id, rowspan, colspan, cell_text, cell_type):
        for i in range(row_id, min(row_id + rowspan, self.max_rows)):
            for j in range(col_id, min(col_id + colspan, self.max_cols)):
                self.content_matrix[i][j] = cell_text
                self.type_matrix[i][j] = cell_type
                self.occupied_cells.add((i, j))




def process_table_trees(tables, table_description):
    table_trees = []
    for table_id, table_html in enumerate(tables):
        table_tree = TableTree(table_html, table_id, table_description)
        table_trees.append(table_tree)
    return table_trees
