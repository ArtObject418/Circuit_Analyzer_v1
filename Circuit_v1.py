from copy import copy
from time import time
from decimal import *

from Vector_class import Vector


def time_check(func):
    def wrap(*args):
        st = time()
        ans = func(*args)
        end = time()
        print(f'Время выполнения {func.__name__}: {end - st:.6f} сек')
        return ans

    return wrap


def matrix_of_equations(vars: list[tuple], equations: list[tuple]):
    matrix = []
    count_vars = len(vars)
    count_equations = len(equations)
    ans = dict.fromkeys(vars, 0.0)
    for equation in equations:
        line = []
        for var in vars:
            if var in equation:
                line.append(equation[equation.index(var) - 1])
                continue
            line.append(0)
        line.append(equation[-1])
        matrix.append(line)
    for iteration in range(count_vars):
        if not matrix[iteration][iteration]:
            for i in range(iteration + 1, count_equations):
                if matrix[i][iteration]:
                    matrix[iteration], matrix[i] = matrix[i], matrix[iteration]
                    break
            else:
                continue
        if matrix[iteration][iteration] != 1:
            matrix[iteration] = [float(Decimal(str(el/matrix[iteration][iteration])).quantize(Decimal('1.000'))) if el != 0 else 0 for el in matrix[iteration]]
        line1 = matrix[iteration]
        main_number = line1[iteration]
        for i in range(count_equations):
            if i == iteration:
                continue
            line2 = matrix[i]
            if line2[iteration] != 0:
                secondary_number = line2[iteration]
                main_line = [secondary_number * _ for _ in line1]
                secondary_line = [main_number * _ for _ in line2]
                matrix[i] = [y - x for x, y in zip(main_line, secondary_line)]

    for iteration in range(count_vars):
        b = matrix[iteration][-1]
        a = matrix[iteration][iteration]
        if a == 0:
            ans[vars[iteration]] = None
            continue
        c = float(Decimal(str(b/a)).quantize(Decimal('1.000')))
        ans[vars[iteration]] = c
    return ans


class CircuitError(Exception):
    def __init__(self, message):
        super().__init__(message)


class Circuit_Block_Info:
    form: str
    outputs_vectors: list[Vector]
    material: str
    specific_resistance: float

    circuit_cords: tuple[int, int] = None
    standard_material: str = 'empty'

    routputs_vectors_dict: dict[Vector, str] = {Vector(0, 0): '0',
                                                Vector(0, 1): 'D',
                                                Vector(0, -1): 'U',
                                                Vector(1, 0): 'R',
                                                Vector(-1, 0): 'L',
                                                }
    outputs_vectors_dict: dict[str, Vector] = {'0': Vector(0, 0),
                                               'D': Vector(0, 1),
                                               'U': Vector(0, -1),
                                               'R': Vector(1, 0),
                                               'L': Vector(-1, 0)}
    materials: dict[str, float] = {'empty': 0.0,
                                   'silver': 0.016,
                                   'copper': 0.017,
                                   'gold': 0.024,
                                   'aluminum': 0.028,
                                   'tungsten': 0.055,
                                   'iron': 0.1,
                                   'lead': 0.21,
                                   'nickeline': 0.4,
                                   'marganine': 0.43,
                                   'constantan': 0.5,
                                   'mercury': 0.98,
                                   'singular': 1.0,
                                   'nichrome': 1.1,
                                   'fechral': 1.3,
                                   'graphite': 13,
                                   'hundred': 100,
                                   'porcelain': 10000000000000000000,
                                   'ebonite': 100000000000000000000,
                                   'infinity': float('inf')
                                   }

    def __init__(self, form, material):
        if form == 'hor':
            self.form = 'RL'
        elif form == 'vert':
            self.form = 'UD'
        else:
            self.form = form
        if self.form[0] == '0':
            self.outputs_vectors = [self.outputs_vectors_dict[self.form[1]]]
        else:
            self.outputs_vectors = [self.outputs_vectors_dict[output] for output in self.form]
        self.material = material
        self.specific_resistance = self.materials[material]

    def get_pos(self):
        return self.circuit_cords


class Wire(Circuit_Block_Info):
    local_resistance: float
    local_amperage: float = 0.0

    def __init__(self, form, material=Circuit_Block_Info.standard_material):
        super().__init__(form, material)
        self.local_resistance = 1 * 0.01 * self.specific_resistance

    def valid_region_check(self, input_vector: Vector):
        if -input_vector in self.outputs_vectors:
            output_vector = self.outputs_vectors - -input_vector
            return output_vector, False
        raise CircuitError('SimpleWire не подключён к предыдущему элементу')

    def get_serial_region_outputs(self, input_vector: Vector):
        if -input_vector in self.outputs_vectors:
            output_vector = self.outputs_vectors - -input_vector
            return output_vector[0]
        raise CircuitError('SimpleWire не подключён к предыдущему элементу')

    def set_standard(self):
        self.local_amperage = 0.0

    def get_info(self):
        return {'type': type(self),
                'material': self.material,
                'form': self.form,
                'local_resistance': self.local_resistance,
                'local_amperage': self.local_amperage
                }


class Resistor(Wire):
    resistor_resistance: float

    def __init__(self, form, resistance=100, material=Circuit_Block_Info.standard_material):
        super().__init__(form, material)
        self.resistor_resistance = resistance
        self.local_resistance = self.local_resistance + self.resistor_resistance

    def get_info(self):
        return {'type': type(self),
                'material': self.material,
                'form': self.form,
                'resistor_resistance': self.resistor_resistance,
                'local_resistance': self.local_resistance,
                'local_amperage': self.local_amperage
                }


class Ammeter(Wire):
    def __init__(self, form, material='empty'):
        super().__init__(form, material)

    def get_amperage(self):
        return self.local_amperage


class Voltmeter(Wire):
    voltage: float = 0.0

    def __init__(self, form, material='infinity'):
        super().__init__(form, material)

    def get_voltage(self):
        return self.voltage

    def set_standard(self):
        self.local_amperage = 0.0
        self.voltage = 0.0

    def get_info(self):
        return {'type': type(self),
                'material': self.material,
                'form': self.form,
                'local_resistance': self.local_resistance,
                'local_amperage': self.local_amperage,
                'voltage': self.voltage
                }


class Global_Node(Circuit_Block_Info):
    local_resistance: float
    outputs_amperages: dict[str, dict[str, float | int | None]]

    def __init__(self, form, material=Circuit_Block_Info.standard_material, potential=0):
        super().__init__(form, material)
        self.local_potential = potential
        self.local_resistance = self.output_resistance = 1 * 0.005 * self.specific_resistance
        self.outputs_amperages = {self.form[1]: 0.0}

    def valid_region_check(self, input_vector: Vector):
        if input_vector == Vector(0, 0):
            return self.outputs_vectors, False
        if -input_vector in self.outputs_vectors:
            return [Vector(0, 0)], True
        raise CircuitError('Global_Node не подключён к предыдущему элементу')

    def get_serial_region_outputs(self, input_vector: Vector):
        if -input_vector in self.outputs_vectors:
            return True
        return False

    def set_standard(self):
        self.outputs_amperages = {self.form[1]: 0.0}

    def get_info(self):
        return {'type': type(self),
                'material': self.material,
                'form': self.form,
                'local_potential': self.local_potential,
                'local_resistance': self.local_resistance,
                'outputs_amperages': self.outputs_amperages
                }


class Local_Node(Circuit_Block_Info):
    outputs_amperages: dict[str, dict[str, float | int | None]]
    local_resistance: float

    local_potential: float = None

    def __init__(self, form, material=Circuit_Block_Info.standard_material):
        super().__init__(form, material)
        self.local_resistance = 1 * 0.005 * self.specific_resistance
        self.outputs_amperages = {output: 0.0 for output in self.form}

    def valid_region_check(self, input_vector: Vector):
        if -input_vector in self.outputs_vectors:
            output_vector = self.outputs_vectors - -input_vector
            return output_vector, False
        raise CircuitError('Local_Node не подключён к предыдущему элементу')

    def get_serial_region_outputs(self, input_vector: Vector):
        if -input_vector in self.outputs_vectors:
            output_vector = self.outputs_vectors - -input_vector
            return output_vector
        raise CircuitError('Local_Node не подключён к предыдущему элементу')

    def set_standard(self):
        self.outputs_amperages = {output: 0.0 for output in self.form}
        self.local_potential = None

    def get_info(self):
        return {'type': type(self),
                'material': self.material,
                'form': self.form,
                'local_potential': self.local_potential,
                'local_resistance': self.local_resistance,
                'outputs_amperages': self.outputs_amperages
                }


class Circuit:
    network: list[list[None | Local_Node | Global_Node | Wire] | (None | Local_Node | Global_Node | Wire)]
    network_size: tuple[int, int]
    global_nodes: dict[tuple[int, int], float]
    local_cords: list[tuple[int, int]]

    serial_regions: list[tuple] = None

    def __init__(self, network):
        self.network = network
        self.network_size = (len(network[0]), len(network))
        self.update()

    def update(self):
        self.__set_standard()
        self.__set_cords_of_elements()
        network = self.network
        self.global_nodes = {(x, y): self.get_block((x, y)).local_potential for y in range(len(network)) for x in
                             range(len(network[0])) if
                             isinstance(network[y][x], Global_Node)}
        self.local_cords = list((x, y) for y in range(len(network)) for x in range(len(network[0])) if
                                isinstance(network[y][x], Local_Node))
        if self._circuit_is_valid():
            self.serial_regions = self.__set_serial_regions()
            self.__set_stats_of_blocks()

    def __set_standard(self):
        self.serial_regions = None
        for line in self.network:
            for block in line:
                if block is None:
                    continue
                block.set_standard()

    def __set_serial_regions(self):
        serial_regions = []
        for global_node_cord in self.global_nodes:
            block: Global_Node = self.get_block(global_node_cord)
            start_vector = block.outputs_vectors[0]
            serial_region = self.serial_region(global_node_cord, start_vector)
            if serial_region is None:
                continue
            for part in serial_region:
                if part in serial_regions or part[::-1] in serial_regions:
                    continue
                serial_regions.append(part)
        return serial_regions

    def serial_region(self, start_pos: tuple[int, int], start_vector: Vector, input_visiting=None):
        if input_visiting is None:
            visiting = []
        else:
            visiting = copy(input_visiting)
        regions = []
        output_name1 = Circuit_Block_Info.routputs_vectors_dict[start_vector]
        pos1 = (start_pos, output_name1)
        current_pos = start_pos + start_vector
        input_vector = start_vector
        visiting.append(start_pos)
        try:
            while True:
                if current_pos in visiting:
                    return None
                if not self.__is_valid_position(current_pos):
                    return None

                block: Wire | Local_Node | Global_Node | None = self.get_block(current_pos)

                if block is None:
                    return None

                if isinstance(block, Global_Node):
                    if block.get_serial_region_outputs(input_vector):
                        output_name2 = Circuit_Block_Info.routputs_vectors_dict[-input_vector]
                        pos2 = (current_pos, output_name2)
                        part = (pos1, pos2)
                        return [part]
                    return None

                if isinstance(block, Local_Node):
                    output_name2 = Circuit_Block_Info.routputs_vectors_dict[-input_vector]
                    pos2 = (current_pos, output_name2)
                    part = (pos1, pos2)
                    regions.append(part)

                    output_vectors = block.get_serial_region_outputs(input_vector)
                    trig = False
                    for output_vector in output_vectors:
                        reg = self.serial_region(current_pos, output_vector, visiting)
                        if reg is None:
                            continue
                        for r in reg:
                            trig = True
                            if r in regions or r[::-1] in regions:
                                continue
                            regions.append(r)
                    if trig:
                        return regions
                    return None

                input_vector = block.get_serial_region_outputs(input_vector)
                current_pos = current_pos + input_vector
        except Exception as e:
            print(f'{current_pos}, ({input_vector.x}, {input_vector.y}) {e}')

    def __set_cords_of_elements(self):
        for y, line in enumerate(self.network):
            for x, element in enumerate(line):
                if element is None:
                    continue
                element.circuit_cords = (x, y)

    def _circuit_is_valid(self):
        for global_node_cord in self.global_nodes:
            if self.__circuit_check(global_node_cord):
                return True
        return False

    def __is_valid_position(self, pos):
        n = self.network_size
        if 0 <= pos[0] <= n[0] - 1 and 0 <= pos[1] <= n[1] - 1:
            return True
        return False

    def get_block(self, pos):
        if self.__is_valid_position(pos):
            return self.network[pos[1]][pos[0]]
        return None

    def __circuit_check(self, global_cord: tuple[int, int]):
        visiting = []
        tasks = [(global_cord, Vector(0, 0))]
        while tasks:
            current_pos, input_vector = tasks.pop()
            if current_pos in visiting:
                continue

            visiting.append(current_pos)

            if not self.__is_valid_position(current_pos):
                continue

            block: Wire | Local_Node | Global_Node | None = self.get_block(current_pos)

            if block is None:
                continue
            try:
                output_vectors, is_end = block.valid_region_check(input_vector)
                if is_end:
                    return True

                for vector in output_vectors:
                    next_pos = current_pos + vector
                    tasks.append((next_pos, vector))
            except CircuitError:
                continue
            except Exception as e:
                print(f'pos: {current_pos}, input_vector: {input_vector.get_cords()}, exception: {e}')
        return False

    def __get_region_resistance(self, region):
        pos1 = region[0][0]
        pos2 = region[1][0]
        input_vector = Circuit_Block_Info.outputs_vectors_dict[region[0][1]]
        region_resistance = self.get_block(pos1).local_resistance
        current_pos = pos1 + input_vector
        while True:
            block = self.get_block(current_pos)
            region_resistance += block.local_resistance

            if current_pos == pos2:
                break

            input_vector = block.get_serial_region_outputs(input_vector)
            current_pos = current_pos + input_vector
        return region_resistance

    def __set_region_stats(self, region, I):
        pos1, output1 = region[0][0], region[0][1]
        pos2, output2 = region[1][0], region[1][1]
        start_block: Global_Node | Local_Node = self.get_block(pos1)
        start_block.outputs_amperages[output1] = I
        end_block: Global_Node | Local_Node = self.get_block(pos2)
        end_block.outputs_amperages[output2] = -I
        v = Circuit_Block_Info.outputs_vectors_dict[output1]
        current_pos = pos1 + v
        while True:
            if current_pos == pos2:
                break

            block: Wire | Ammeter | Voltmeter = self.get_block(current_pos)
            if isinstance(block, Voltmeter):
                block.voltage = abs(float(
                    Decimal(str(start_block.local_potential - end_block.local_potential)).quantize(Decimal('1.000'))))
            block.local_amperage = abs(I)
            v = block.get_serial_region_outputs(v)
            current_pos = current_pos + v

    def __set_stats_of_blocks(self):
        if len(self.serial_regions) == 1:
            serial_region = self.serial_regions[0]
            r = self.__get_region_resistance(serial_region)
            pos1, output1 = serial_region[0][0], serial_region[0][1]
            pos2, output2 = serial_region[1][0], serial_region[1][1]
            block1 = self.get_block(pos1)
            block2 = self.get_block(pos2)
            p1 = block1.local_potential
            p2 = block2.local_potential
            if p1 > p2:
                I = (p1 - p2) / r
                self.__set_region_stats(serial_region, I)
                return None
            elif p1 == p2:
                return None
            else:
                I = (p2 - p1) / r
                self.__set_region_stats(serial_region[::-1], I)
                return None

        vars = []
        equations = []
        local_nodes: dict[tuple, list[tuple]] = {}
        serial_regions_vars: dict[tuple, tuple] = {}
        for serial_region in self.serial_regions:
            r = self.__get_region_resistance(serial_region)
            if r == float('inf'):
                serial_regions_vars[serial_region] = None
                continue
            pos1, output1 = serial_region[0][0], serial_region[0][1]
            pos2, output2 = serial_region[1][0], serial_region[1][1]
            block1 = self.get_block(pos1)
            block2 = self.get_block(pos2)
            if isinstance(block1, Global_Node):
                region_vars = ((pos2, 'P'), (pos1, 'I', output1), (pos2, 'I', output2))
                region_equations = ((r, region_vars[1], 1, region_vars[0], block1.local_potential),
                                    (-r, region_vars[2], 1, region_vars[0], block1.local_potential),
                                    (1, region_vars[1], 1, region_vars[2], 0)
                                    )
                local_nodes[pos2] = [1, region_vars[2]] + local_nodes.get(pos2, [0])
                for var in region_vars:
                    if var in vars:
                        continue
                    vars.append(var)
                for region_equation in region_equations:
                    if region_equation in equations:
                        continue
                    equations.append(region_equation)

            elif isinstance(block2, Global_Node):
                region_vars = ((pos1, 'P'), (pos1, 'I', output1), (pos2, 'I', output2))
                region_equations = ((r, region_vars[2], 1, region_vars[0], block2.local_potential),
                                    (-r, region_vars[1], 1, region_vars[0], block2.local_potential),
                                    (1, region_vars[1], 1, region_vars[2], 0)
                                    )
                local_nodes[pos1] = [1, region_vars[1]] + local_nodes.get(pos1, [0])
                for var in region_vars:
                    if var in vars:
                        continue
                    vars.append(var)
                for region_equation in region_equations:
                    if region_equation in equations:
                        continue
                    equations.append(region_equation)

            else:
                region_vars = ((pos1, 'P'), (pos2, 'P'), (pos1, 'I', output1), (pos2, 'I', output2))
                region_equations = ((1, region_vars[0], -1, region_vars[1], -r, region_vars[2], 0),
                                    (-1, region_vars[0], 1, region_vars[1], -r, region_vars[3], 0),
                                    (1, region_vars[2], 1, region_vars[3], 0)
                                    )
                local_nodes[pos1] = [1, region_vars[2]] + local_nodes.get(pos1, [0])
                local_nodes[pos2] = [1, region_vars[3]] + local_nodes.get(pos2, [0])
                for var in region_vars:
                    if var in vars:
                        continue
                    vars.append(var)
                for region_equation in region_equations:
                    if region_equation in equations:
                        continue
                    equations.append(region_equation)
            serial_regions_vars[serial_region] = region_vars
        for local_node, outputs in local_nodes.items():
            equation = tuple(outputs)
            if equation in equations:
                continue
            equations.append(equation)
        stats = matrix_of_equations(vars, equations)
        for var in stats:
            if var[1] == 'P':
                block: Local_Node = self.get_block(var[0])
                block.local_potential = stats[var]
        for serial_region in serial_regions_vars:
            if serial_regions_vars[serial_region] is None:
                self.__set_region_stats(serial_region, 0.0)
                continue
            values = {var: stats[var] for var in serial_regions_vars[serial_region]}
            pos1, output1 = serial_region[0][0], serial_region[0][1]
            pos2, output2 = serial_region[1][0], serial_region[1][1]
            block1 = self.get_block(pos1)
            block2 = self.get_block(pos2)
            if isinstance(block1, Global_Node):
                if block1.local_potential > values[(pos2, 'P')]:
                    self.__set_region_stats(serial_region, values[(pos1, 'I', output1)])
                elif block1.local_potential < values[(pos2, 'P')]:
                    self.__set_region_stats(serial_region[::-1], values[(pos2, 'I', output2)])
                else:
                    if values[(pos1, 'I', output1)] > values[(pos2, 'I', output2)]:
                        self.__set_region_stats(serial_region, values[(pos1, 'I', output1)])
                    elif values[(pos1, 'I', output1)] < values[(pos2, 'I', output2)]:
                        self.__set_region_stats(serial_region[::-1], values[(pos2, 'I', output2)])
                    elif values[(pos1, 'I', output1)] == values[(pos1, 'I', output1)] == 0:
                        self.__set_region_stats(serial_region, 0.0)
                    else:
                        raise CircuitError('Ошибка в расставлении параметров участка с 0 напряжением')
            elif isinstance(block2, Global_Node):
                if block2.local_potential > values[(pos1, 'P')]:
                    self.__set_region_stats(serial_region[::-1], values[(pos2, 'I', output2)])
                elif block2.local_potential < values[(pos1, 'P')]:
                    self.__set_region_stats(serial_region, values[(pos1, 'I', output1)])
                else:
                    if values[(pos1, 'I', output1)] > values[(pos2, 'I', output2)]:
                        self.__set_region_stats(serial_region, values[(pos1, 'I', output1)])
                    elif values[(pos1, 'I', output1)] < values[(pos2, 'I', output2)]:
                        self.__set_region_stats(serial_region[::-1], values[(pos2, 'I', output2)])
                    elif values[(pos1, 'I', output1)] == values[(pos1, 'I', output1)] == 0:
                        self.__set_region_stats(serial_region, 0.0)
                    else:
                        raise CircuitError('Ошибка в расставлении параметров участка с 0 напряжением')
            else:
                if values[(pos1, 'I', output1)] > values[(pos2, 'I', output2)]:
                    self.__set_region_stats(serial_region, values[(pos1, 'I', output1)])
                elif values[(pos1, 'I', output1)] < values[(pos2, 'I', output2)]:
                    self.__set_region_stats(serial_region[::-1], values[(pos2, 'I', output2)])
                elif values[(pos1, 'I', output1)] == values[(pos1, 'I', output1)] == 0:
                    self.__set_region_stats(serial_region, 0.0)
                else:
                    raise CircuitError('Ошибка в расставлении параметров участка с 0 напряжением')
        return None


if __name__ == '__main__':
    # vars = ['x', 'y', 'z']
    # equations = [(1, 'x', 10, 'y', 41),
    #              (1, 'z', -10, 'y', -43.5),
    #              (1, 'x', 0.5, 'y', 1, 'z', -0.5),
    #              (1, 'x', -1, 'z', 4.5)]
    # print(matrix_of_equations(vars, equations))
    # 1
    # net = [[Global_Node('0R', potential=10), Local_Node('RDL'), Wire('hor'), Local_Node('RDL'), Global_Node('0L')],
    #        [Wire('RD'), Local_Node('DLU'), None, Wire('vert'), None],
    #        [Wire('vert'), Wire('vert'), None, Wire('vert'), None],
    #        [Wire('RU'), Local_Node('LUR'), Wire('hor'), Wire('LU'), None],
    #        [None, None, None, None, None]
    #        ]
    # Net = Circuit(net)
    # print(next(it), Net._circuit_is_valid())
    # print(Net.serial_regions)
    # net = [[Global_Node('0R', potential=10), Wire('LD'), Global_Node('0D', potential=1)],
    #        [None, Wire('vert'), Wire('vert')],
    #        [None, Wire('RU'), Wire('LU')]
    #        ]
    # Net = Circuit(net)
    # print(next(it), Net._circuit_is_valid())
    # print(Net.serial_regions)
    #
    # net = [[Global_Node('0R', potential=10), Local_Node('RUL'), Global_Node('0L', potential=0)],
    #        [None, Wire('vert'), Wire('vert')],
    #        [None, Wire('RU'), Wire('LU')]
    #        ]
    #
    # net = [[Global_Node('0R', potential=10), Local_Node('RDL'), Local_Node('RDL'), Global_Node('0L', potential=1)],
    #        [None, Voltmeter('vert'), Wire('vert'), None],
    #        [None, Wire('RU'), Wire('LU'), None]
    #        ]
    # # 2
    # net = [[Global_Node('0R'), Global_Node('0R')]
    #        ]
    # Net = Circuit(net)
    # print(next(it), Net._circuit_is_valid())
    # print(Net.serial_regions)
    #
    # 3
    # net = [[Global_Node('0R', potential=2),
    #         Wire('hor'),
    #         Global_Node('0L', potential=0)]
    #        ]
    #
    # # 4
    # net = [[Global_Node('0R'), Wire('vert'), Global_Node('0L')]
    #        ]
    # Net = Circuit(net)
    # print(next(it), Net._circuit_is_valid())
    # print(Net.serial_regions)
    #
    # 5
    # net = [[Global_Node('0R'), Wire('hor'), Global_Node('0L')]
    #        ]
    # Net = Circuit(net)
    # print(next(it), Net._circuit_is_valid())
    # print(Net.serial_regions)
    #
    # 6
    # net = [[Global_Node('0R', potential=10), Wire('hor'), Local_Node('RDL'),
    #         Global_Node('0L', potential=1)]
    #        ]
    # Net = Circuit(net)
    # print(next(it), Net._circuit_is_valid())
    # print(Net.serial_regions)
    #
    # #7
    # net = [[Global_Node('0R', potential=10), Local_Node('RDL'), Local_Node('RDL'), Global_Node('0L', potential=0), None,
    #         None],
    #        [None, Resistor('vert'), Wire('vert'), Wire('RD'), Wire('hor'), None],
    #        [Wire('RD'), Wire('LU'), Local_Node('URD'), Local_Node('RDLU'), Local_Node('RDL'), Wire('hor')],
    #        [Local_Node('URD'), Wire('LD'), Wire('vert'), Global_Node('0U', potential=6), Wire('vert'), None],
    #        [Wire('vert'), Wire('RU'), Wire('LU'), None, None, None]
    #        ]
    # Net = Circuit(net)
    # print(next(it), Net._circuit_is_valid())
    # print([('(0, 0)R', '(1, 0)L'), ('(3, 0)L', '(2, 0)R'), ('(3, 3)U', '(3, 2)D'), ('(1, 0)R', '(2, 0)L'),
    #        ('(1, 0)D', '(0, 3)U'), ('(2, 0)D', '(2, 2)U'), ('(2, 2)R', '(3, 2)L'), ('(2, 2)D', '(0, 3)R'),
    #        ('(3, 2)R', '(4, 2)L')])
    # print(Net.serial_regions)
    # # '''d = Local_Node('RDL')
    # # print(d.outputs_amperages)'''

    # net = [[Global_Node('0R', potential=1000), Local_Node('RDL'), Ammeter('hor'), Resistor('hor', resistance=100), Local_Node('RDL'), Global_Node('0L', potential=0)],
    #        [None, Local_Node('URD'), Ammeter('hor'), Voltmeter('hor'), Local_Node('RULD'), Wire('LD')]]
    # net = [[Global_Node('0R', potential=10), Voltmeter('hor'), Global_Node('0L', potential=0)]]
    net = [[Global_Node('0R', potential=1000), Local_Node('RDL'), Ammeter('hor'), Resistor('hor', resistance=100), Local_Node('RDL'), Global_Node('0L', potential=10)],
           [None, Local_Node('URD'), Ammeter('hor'), Voltmeter('hor'), Local_Node('RULD'), Wire('LD')],
           [None, Ammeter('vert'), None, None, Resistor('vert', resistance=100), Voltmeter('vert')],
           [None, Resistor('vert', resistance=100), None, None, Local_Node('URD'), Wire('LU')],
           [Global_Node('0R', potential=200), Local_Node('LUR'), Ammeter('hor'), Wire('hor'), Wire('LU'), None],
           [None, None, None, None, None, None]
           ]
    st = time()
    Net = Circuit(net)
    print('',
          f'(2, 0): A = {Net.get_block((2, 0)).local_amperage}\n',
          f'(2, 1): A = {Net.get_block((2, 1)).local_amperage}\n',
          f'(3, 1): V = {Net.get_block((3, 1)).voltage}\n',
          f'(1, 2): A = {Net.get_block((1, 2)).local_amperage}\n',
          f'(5, 2): V = {Net.get_block((5, 2)).voltage}\n',
          f'(2, 4): A = {Net.get_block((2, 4)).local_amperage}\n')
    for y in range(Net.network_size[1]):
        print(y)
        for x in range(Net.network_size[0]):
            block = Net.get_block((x, y))
            if block is None:
                print('\ttype: None')
                continue
            print(f'\t{block.get_info()}')
        print()
    end = time()
    print(end-st)
