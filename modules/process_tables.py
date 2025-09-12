import json
from bs4 import BeautifulSoup
from typing import Dict, List, Set, Tuple, Any
import os


class TableMatrixParser:
    def __init__(self):
        self.content_matrix: List[List[str]] = []  # 存储单元格内容
        self.type_matrix: List[List[str]] = []  # 存储单元格类型
        self.merge_counter: int = 0  # 合并单元格计数器
        self.max_rows: int = 0
        self.max_cols: int = 0
        self.occupied_cells: Set[Tuple[int, int]] = set()  # 记录已被占用的单元格位置
        self.header_row_boundary: int = 0  # 表头行的边界
        self.header_col_boundary: int = 0  # 表头列的边界

    def initialize_matrices(self, rows: int, cols: int) -> None:
        self.max_rows = rows
        self.max_cols = cols
        self.content_matrix = [['' for _ in range(cols)] for _ in range(rows)]
        self.type_matrix = [['' for _ in range(cols)] for _ in range(rows)]
        self.occupied_cells.clear()

    # 计算表格的最大尺寸
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

    def is_data_cell(self, row_idx: int, col_idx: int, table_id: int, table_description: Dict) -> bool:
        cell_key = f"{table_id}-{row_idx}-{col_idx}"
        return cell_key in table_description

    def _determine_cell_type(self, cell: Any, row_idx: int, col_idx: int,
                             table_id: int, table_description: Dict) -> str:
        text = cell.get_text(strip=True)

        # 检查是否为空单元格
        if not text or text.strip() in ['']:
            return 'empty'

        # 没有数据也视为数据单元格
        if not text or text.strip() in ['-', '—']:
            return 'data'

        # 检查是否为数据单元格
        if self.is_data_cell(row_idx, col_idx, table_id, table_description):
            return 'data'

        # 检查是否为合并单元格
        if int(cell.get('colspan', 1)) > 1 or int(cell.get('rowspan', 1)) > 1:
            self.merge_counter += 1
            return f'merge_{self.merge_counter}'

        return ''

    def _fill_cell_area(self, row: int, col: int, rowspan: int, colspan: int,
                        content: str, cell_type: str) -> None:
        for i in range(row, min(row + rowspan, self.max_rows)):
            for j in range(col, min(col + colspan, self.max_cols)):
                self.content_matrix[i][j] = content
                self.type_matrix[i][j] = cell_type
                self.occupied_cells.add((i, j))

    # 生成内容矩阵和类型矩阵
    def fill_matrices(self, table_soup: BeautifulSoup, table_id: int, table_description: Dict) -> None:
        rows = table_soup.find_all('tr')

        for row_idx, row in enumerate(rows):
            col_idx = 0
            cells = row.find_all(['td', 'th'])

            for cell in cells:
                # 跳过已被占用的位置
                while (row_idx, col_idx) in self.occupied_cells and col_idx < self.max_cols:
                    col_idx += 1

                if col_idx >= self.max_cols:
                    break

                # 获取单元格属性
                colspan = int(cell.get('colspan', 1))
                rowspan = int(cell.get('rowspan', 1))
                text = cell.get_text(strip=True)

                # 确定单元格类型
                cell_type = self._determine_cell_type(cell, row_idx, col_idx,
                                                      table_id, table_description)

                # 填充区域
                self._fill_cell_area(row_idx, col_idx, rowspan, colspan, text, cell_type)

                col_idx += colspan

    def find_data_boundaries(self) -> Tuple[int, int]:
        """
        找到数据类型首次出现的位置，确定表头边界
        从第二行和第二列开始搜索，跳过可能的表头行和列
        如果没有找到数据单元格，则返回默认值0
        """
        row_first_data = []
        col_first_data = []

        # 检查每一行（从第二行开始）
        for row_idx in range(1, self.max_rows):
            for col_idx in range(1, self.max_cols):
                if self.type_matrix[row_idx][col_idx] == 'data':
                    row_first_data.append(col_idx)
                    break

        # 检查每一列（从第二列开始）
        for col_idx in range(1, self.max_cols):
            for row_idx in range(1, self.max_rows):
                if self.type_matrix[row_idx][col_idx] == 'data':
                    col_first_data.append(row_idx)
                    break

        row_boundary = min(row_first_data) if row_first_data else 0
        col_boundary = min(col_first_data) if col_first_data else 0

        self.header_row_boundary = row_boundary
        self.header_col_boundary = col_boundary

        # 检查边界行是否为一级标题
        if col_boundary > 0:
            # 检查该行是否只有第一列有内容，其他列都为空
            is_first_level_title = True
            for col in range(row_boundary, self.max_cols):
                if not (self.type_matrix[col_boundary - 1][col] == 'empty' 
                        or not self.content_matrix[col_boundary - 1][col].strip()):
                    is_first_level_title = False
                    break

            if is_first_level_title and self.content_matrix[col_boundary - 1][0].strip():
                self.header_col_boundary -= 1
                col_boundary -= 1

        return row_boundary, col_boundary

    def build_column_header_tree(self) -> List[Dict]:

        def create_node(text: str, col_coords: List[int], level_id: int) -> Dict:
            return {
                "level_id": level_id,
                "text": text,
                "col_coords": col_coords,  # 改为列表类型
                "children": []
            }

        def is_merge_cell(cell_type: str) -> bool:
            return cell_type.startswith('merge_')

        def get_merge_cell_width(row: int, col: int, cell_type: str) -> int:
            """获取合并单元格的宽度"""
            width = 0
            curr_col = col
            while (curr_col < self.max_cols and
                   self.type_matrix[row][curr_col] == cell_type):
                width += 1
                curr_col += 1
            return width

        # 初始化结果列表，每列对应一个顶层节点
        headers = []
        processed_cells = set()  # 记录已处理的单元格坐标

        # 首先处理第0行，创建根节点
        for col in range(self.max_cols):
            if (0, col) in processed_cells or self.type_matrix[0][col] == 'empty':
                continue

            cell_type = self.type_matrix[0][col]
            cell_text = self.content_matrix[0][col]

            # 获取所有涉及的列坐标
            col_coords = [col]  # 初始化为当前列
            if is_merge_cell(cell_type):
                # 获取合并单元格的宽度
                merge_width = get_merge_cell_width(0, col, cell_type)
                # 添加所有被合并的列坐标
                col_coords.extend(range(col + 1, col + merge_width))
                # 标记所有被合并的单元格
                for c in range(col, col + merge_width):
                    processed_cells.add((0, c))
            else:
                processed_cells.add((0, col))

            # 创建根节点
            root_node = create_node(
                text=cell_text,
                col_coords=col_coords,
                level_id=0
            )

            headers.append(root_node)

        # 处理后续行，建立父子关系
        for row in range(1, self.header_col_boundary):
            col = 0
            while col < self.max_cols:
                if (row, col) in processed_cells or self.type_matrix[row][col] == 'empty':
                    col += 1
                    continue

                cell_type = self.type_matrix[row][col]
                cell_text = self.content_matrix[row][col]

                # 获取所有涉及的列坐标
                col_coords = [col]  # 初始化为当前列
                cell_width = 1

                if is_merge_cell(cell_type):
                    # 获取合并单元格的宽度
                    cell_width = get_merge_cell_width(row, col, cell_type)
                    # 添加所有被合并的列坐标
                    col_coords.extend(range(col + 1, col + cell_width))

                # 创建当前节点
                current_node = create_node(
                    text=cell_text,
                    col_coords=col_coords,
                    level_id=row
                )

                # 标记处理过的单元格
                for c in range(col, col + cell_width):
                    processed_cells.add((row, c))

                # 找到父节点
                found_parents = []  # 改用列表存储所有可能的父节点
                
                def find_parent_in_children(node, current_row, col_coord) -> bool:
                    """递归检查节点的子节点是否可能是父节点"""
                    if current_row == 0:  # 已经检查到第一行，不需要继续
                        return False
                        
                    for child in node['children']:
                        child_cols = set(child['col_coords'])
                        # 如果当前检查的列坐标在子节点的列坐标范围内
                        if col_coord in child_cols:
                            # 继续检查这个子节点的子节点
                            if find_parent_in_children(child, current_row - 1, col_coord):
                                return True
                            # 如果没有更深的子节点是父节点，那这个子节点就是父节点
                            found_parents.append((child, col_coord))
                            return True
                    return False

                # 遍历每一个列坐标，找到对应的父节点
                for col_coord in col_coords:
                    parent_found_for_col = False
                    # 遍历顶层节点
                    for parent in headers:
                        parent_cols = set(parent['col_coords'])
                        # 检查当前列坐标是否在父节点的列坐标范围内
                        if col_coord in parent_cols:
                            # 先检查这个节点的子节点是否可能是父节点
                            if not find_parent_in_children(parent, row - 1, col_coord):
                                # 如果子节点都不是父节点，那这个顶层节点就是父节点
                                found_parents.append((parent, col_coord))
                            parent_found_for_col = True
                            break
                    
                    # 如果这一列没有找到父节点，需要检查是否有相同内容的顶层节点
                    if not parent_found_for_col:
                        # 检查是否已经有相同内容的顶层节点
                        existing_node = None
                        for node in headers:
                            if node['text'] == cell_text:
                                existing_node = node
                                break
                        
                        if existing_node:
                            # 如果找到相同内容的节点，只需要更新其列坐标
                            existing_node['col_coords'].append(col_coord)
                        else:
                            # 否则创建新的顶层节点
                            new_node = create_node(
                                text=cell_text,
                                col_coords=[col_coord],
                                level_id=row
                            )
                            headers.append(new_node)
                
                # 如果找到了父节点，为每个父节点创建或更新子节点
                if found_parents:
                    # 按父节点分组，将相同父节点的列坐标合并
                    parent_to_cols = {}
                    parent_objects = {}  # 用于存储父节点对象
                    for parent, col_coord in found_parents:
                        parent_id = id(parent)  # 使用对象的id作为键
                        if parent_id not in parent_to_cols:
                            parent_to_cols[parent_id] = []
                            parent_objects[parent_id] = parent
                        parent_to_cols[parent_id].append(col_coord)
                    
                    # 为每个父节点处理子节点
                    for parent_id, cols in parent_to_cols.items():
                        parent = parent_objects[parent_id]
                        # 检查父节点是否已经有相同内容的子节点
                        existing_node = None
                        for child in parent['children']:
                            if child['text'] == cell_text:
                                existing_node = child
                                break
                        
                        if existing_node:
                            # 如果找到相同内容的节点，只需要更新其列坐标
                            existing_node['col_coords'].extend(cols)
                        else:
                            # 否则创建新的子节点
                            child_node = create_node(
                                text=cell_text,
                                col_coords=cols,  # 包含所有对应的列
                                level_id=row
                            )
                            parent['children'].append(child_node)
                
                col += cell_width

        return headers

    def build_row_header_tree(self) -> List[Dict]:
        def create_node(text: str, row_coords: List[int], level_id: int) -> Dict:
            return {
                "level_id": level_id,
                "text": text,
                "row_coords": row_coords,  # 改为列表类型
                "children": []
            }

        def is_empty_cell(cell_type: str, content: str) -> bool:
            return cell_type == 'empty' or not content.strip()

        def is_merge_cell(cell_type: str) -> bool:
            return cell_type.startswith('merge_')

        def get_merge_cell_height(row: int, col: int, cell_type: str) -> int:
            """获取合并单元格的高度"""
            height = 1
            next_row = row + 1
            while (next_row < self.max_rows and
                   self.type_matrix[next_row][col] == cell_type):
                height += 1
                next_row += 1
            return height

        def get_merge_cell_width(row: int, col: int, cell_type: str) -> int:
            """获取合并单元格的宽度"""
            width = 1
            next_col = col + 1
            while (next_col < self.header_row_boundary and
                   self.type_matrix[row][next_col] == cell_type):
                width += 1
                next_col += 1
            return width

        def process_horizontal_headers(start_row: int, end_row: int) -> List[Dict]:
            """处理横向的表头关系，包括合并单元格"""
            headers = []
            processed_cells = set()

            # 从左到右处理每一列
            col = 0
            while col < self.header_row_boundary:
                if (start_row, col) in processed_cells:
                    col += 1
                    continue

                cell_type = self.type_matrix[start_row][col]
                cell_text = self.content_matrix[start_row][col]

                if is_empty_cell(cell_type, cell_text):
                    col += 1
                    continue

                # 获取所有涉及的行坐标
                row_coords = [start_row]  # 初始化为当前行
                merge_height = 1
                merge_width = 1

                # 如果是合并单元格，处理其合并范围
                if is_merge_cell(cell_type):
                    merge_height = get_merge_cell_height(start_row, col, cell_type)
                    merge_width = get_merge_cell_width(start_row, col, cell_type)
                    # 添加所有被合并的行坐标
                    row_coords.extend(range(start_row + 1, start_row + merge_height))

                # 创建当前节点
                current_node = create_node(
                    text=cell_text,
                    row_coords=row_coords,
                    level_id=len(headers)
                )

                # 标记被合并的单元格
                for r in range(start_row, start_row + merge_height):
                    for c in range(col, col + merge_width):
                        processed_cells.add((r, c))

                # 处理右侧的每一行的子节点
                for r in range(start_row, start_row + merge_height):
                    # 从当前单元格的右侧开始处理，考虑colspan
                    next_col = col + merge_width
                    while next_col < self.header_row_boundary:
                        next_cell_type = self.type_matrix[r][next_col]
                        next_cell_text = self.content_matrix[r][next_col]

                        if not is_empty_cell(next_cell_type, next_cell_text) and (r, next_col) not in processed_cells:
                            child_row_coords = [r]
                            child_width = 1
                            
                            if is_merge_cell(next_cell_type):
                                child_height = get_merge_cell_height(r, next_col, next_cell_type)
                                child_width = get_merge_cell_width(r, next_col, next_cell_type)
                                child_row_coords.extend(range(r + 1, r + child_height))
                                # 标记子节点的合并单元格
                                for child_r in range(r, r + child_height):
                                    for child_c in range(next_col, next_col + child_width):
                                        processed_cells.add((child_r, child_c))
                            else:
                                processed_cells.add((r, next_col))

                            child_node = create_node(
                                text=next_cell_text,
                                row_coords=child_row_coords,
                                level_id=len(headers) + 1
                            )
                            current_node['children'].append(child_node)
                            
                            next_col += child_width
                        else:
                            next_col += 1

                headers.append(current_node)
                col += merge_width

            return headers

        # 第一阶段：查找并处理一级标题
        headers = []
        current_first_level = None
        processed_rows = set()

        for row in range(self.header_col_boundary, self.max_rows):
            if row in processed_rows:
                continue

            # 检查第一列的单元格
            cell_type = self.type_matrix[row][0]
            cell_text = self.content_matrix[row][0]

            if is_empty_cell(cell_type, cell_text):
                continue

            # 检查是否为一级标题（右侧单元格均为空）
            is_first_level = all(
                is_empty_cell(self.type_matrix[row][col], self.content_matrix[row][col])
                for col in range(self.header_row_boundary, self.max_cols)
            )

            if is_first_level:
                # 获取所有涉及的行坐标
                row_coords = [row]  # 初始化为当前行
                merge_height = 1

                # 如果是合并单元格，获取所有被合并的行
                if is_merge_cell(cell_type):
                    merge_height = get_merge_cell_height(row, 0, cell_type)
                    row_coords.extend(range(row + 1, row + merge_height))

                # 创建新的一级标题节点
                current_first_level = create_node(
                    text=cell_text,
                    row_coords=row_coords,
                    level_id=len(headers)
                )
                headers.append(current_first_level)

                # 标记被合并的行
                for r in range(row, row + merge_height):
                    processed_rows.add(r)

            elif current_first_level is not None:
                # 处理当前行的横向表头关系，并添加为一级标题的子节点
                row_headers = process_horizontal_headers(row, self.max_rows)
                if row_headers:
                    current_first_level['children'].extend(row_headers)
            else:
                # 如果没有一级标题，直接处理横向关系
                row_headers = process_horizontal_headers(row, self.max_rows)
                headers.extend(row_headers)

        return headers

    def recalculate_col_level_id(self, headers: List[Dict]) -> List[Dict]:
        def recalculate_level(nodes: List[Dict], depth: int) -> None:
            if not nodes:
                return
                
            # 按照 col_coords 中最小的值排序
            sorted_nodes = sorted(nodes, key=lambda x: min(x['col_coords']))
            
            # 更新当前层级的 level_id
            for i, node in enumerate(sorted_nodes):
                node['level_id'] = i
                # 递归处理子节点
                recalculate_level(node['children'], depth + 1)
        
        # 从根节点开始重新计算
        recalculate_level(headers, 0)
        return headers

    def recalculate_row_level_id(self, headers: List[Dict]) -> List[Dict]:
        def recalculate_level(nodes: List[Dict], depth: int) -> None:
            if not nodes:
                return
                
            # 按照 row_coords 中最小的值排序
            sorted_nodes = sorted(nodes, key=lambda x: min(x['row_coords']))
            
            # 更新当前层级的 level_id
            for i, node in enumerate(sorted_nodes):
                node['level_id'] = i
                # 递归处理子节点
                recalculate_level(node['children'], depth + 1)
        
        # 从根节点开始重新计算
        recalculate_level(headers, 0)
        return headers


def process_table_tree(table_html, tid, table_description):
    parser = TableMatrixParser()
    soup = BeautifulSoup(table_html, 'html.parser')

    # 计算表格尺寸并初始化矩阵
    max_rows, max_cols = parser.calculate_table_dimensions(soup)
    parser.initialize_matrices(max_rows, max_cols)

    # 填充矩阵
    parser.fill_matrices(soup, tid, table_description)

    # 找到表头边界
    row_boundary, col_boundary = parser.find_data_boundaries()

    # 构建列表头树和行表头树
    col_headers = parser.build_column_header_tree()
    row_headers = parser.build_row_header_tree()

    # 重新计算level_id: 定义level_id在一个层级中，是从0开始，在层级中，level_id是唯一的，在不同层级中，level_id可以相同
    # 且根据col_coords或者row_coords的值，越小，level_id越小

    col_headers = parser.recalculate_col_level_id(col_headers)
    row_headers = parser.recalculate_row_level_id(row_headers)

    return {
        "table_id": tid,
        "table_html": table_html,
        "content_matrix": parser.content_matrix,
        "type_matrix": parser.type_matrix,
        "row_boundary": row_boundary,
        "col_boundary": col_boundary,
        "col_headers": col_headers,
        "row_headers": row_headers
    }


def process_table_trees(tables, table_description):
    table_trees = []
    for tid, table_html in enumerate(tables):
        table_tree = process_table_tree(table_html, tid, table_description)
        table_trees.append(table_tree)
    return table_trees
