from bs4 import BeautifulSoup
from typing import List, Tuple, Set

class TableMatrixParser:
    def __init__(self):
        self.content_matrix: List[List[str]] = []
        self.type_matrix: List[List[str]] = []
        self.merge_counter: int = 0
        self.max_rows: int = 0
        self.max_cols: int = 0
        self.occupied_cells: Set[Tuple[int, int]] = set()

    def initialize_matrices(self, rows: int, cols: int) -> None:
        self.max_rows = rows
        self.max_cols = cols
        self.content_matrix = [['' for _ in range(cols)] for _ in range(rows)]
        self.type_matrix = [['' for _ in range(cols)] for _ in range(rows)]
        self.occupied_cells.clear()

    def calculate_table_dimensions(self, table_soup: BeautifulSoup) -> Tuple[int, int]:
        rows = table_soup.find_all('tr')
        max_rows = len(rows)
        max_cols = 0

        for row in rows:
            current_col = 0
            for cell in row.find_all(['td', 'th']):
                colspan = int(cell.get('colspan', 1))
                current_col += colspan
            max_cols = max(max_cols, current_col)

        return max_rows, max_cols

    def _determine_cell_type(self, cell) -> str:
        text = cell.get_text(strip=True)
        
        if not text:
            return 'empty'
            
        if int(cell.get('colspan', 1)) > 1 or int(cell.get('rowspan', 1)) > 1:
            self.merge_counter += 1
            return f'merge_{self.merge_counter}'
            
        return 'data'

    def _fill_cell_area(self, row: int, col: int, rowspan: int, colspan: int,
                       content: str, cell_type: str) -> None:
        for i in range(row, min(row + rowspan, self.max_rows)):
            for j in range(col, min(col + colspan, self.max_cols)):
                self.content_matrix[i][j] = content
                self.type_matrix[i][j] = cell_type
                self.occupied_cells.add((i, j))

    def fill_matrices(self, table_soup: BeautifulSoup) -> None:
        rows = table_soup.find_all('tr')
        
        for row_idx, row in enumerate(rows):
            col_idx = 0
            cells = row.find_all(['td', 'th'])
            
            for cell in cells:
                while (row_idx, col_idx) in self.occupied_cells and col_idx < self.max_cols:
                    col_idx += 1
                    
                if col_idx >= self.max_cols:
                    break
                    
                colspan = int(cell.get('colspan', 1))
                rowspan = int(cell.get('rowspan', 1))
                text = cell.get_text(strip=True)
                
                cell_type = self._determine_cell_type(cell)
                
                self._fill_cell_area(row_idx, col_idx, rowspan, colspan, text, cell_type)
                
                col_idx += colspan

    def extract_submatrix(self, row_indices: List[int], col_indices: List[int]) -> Tuple[List[List[str]], List[List[str]]]:
        sub_content = [[self.content_matrix[i][j] for j in col_indices] for i in row_indices]
        sub_type = [[self.type_matrix[i][j] for j in col_indices] for i in row_indices]
        return sub_content, sub_type

    def restore_html(self, content_matrix: List[List[str]], type_matrix: List[List[str]]) -> str:
        rows = len(content_matrix)
        cols = len(content_matrix[0]) if rows > 0 else 0
        
        soup = BeautifulSoup('', 'html.parser')
        table = soup.new_tag('table')
        processed = set()
        
        for i in range(rows):
            tr = soup.new_tag('tr')
            j = 0
            while j < cols:
                if (i, j) in processed:
                    j += 1
                    continue
                    
                content = content_matrix[i][j]
                cell_type = type_matrix[i][j]
                
                td = soup.new_tag('td')
                td.string = content
                
                if cell_type.startswith('merge_'):
                    colspan = 1
                    while j + colspan < cols and type_matrix[i][j + colspan] == cell_type:
                        colspan += 1
                        
                    rowspan = 1
                    while i + rowspan < rows and type_matrix[i + rowspan][j] == cell_type:
                        rowspan += 1
                        
                    for r in range(i, i + rowspan):
                        for c in range(j, j + colspan):
                            processed.add((r, c))
                            
                    if colspan > 1:
                        td['colspan'] = str(colspan)
                    if rowspan > 1:
                        td['rowspan'] = str(rowspan)
                else:
                    processed.add((i, j))
                
                tr.append(td)
                j += 1
                
            table.append(tr)
            
        return str(table)

def extract_subtable(html: str, row_indices: List[int], col_indices: List[int]) -> str:
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table')
    
    if not table:
        raise ValueError("None")

    parser = TableMatrixParser()
    max_rows, max_cols = parser.calculate_table_dimensions(table)
    parser.initialize_matrices(max_rows, max_cols)
    parser.fill_matrices(table)
    sub_content, sub_type = parser.extract_submatrix(row_indices, col_indices)
    
    return parser.restore_html(sub_content, sub_type)