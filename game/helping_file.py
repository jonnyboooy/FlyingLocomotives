import math


class Geometry:
    pass


class Line(Geometry):
    def __init__(self, point1: tuple[float, float], point2: tuple[float, float]):
        self.point1 = point1
        self.point2 = point2

        self.line_by_coordinates = (self.point1, self.point2)
        self.median_point = ((self.point1[0] + self.point2[0]) / 2, (self.point1[1] + self.point2[1]) / 2)

        self.line = self.line_equation_calculations
        self.median_perpendicular = self.median_perpendicular_equation_calculations

    @property
    def points_of_line(self) -> tuple:
        '''
        Возвращает кортеж двух точек (point_1, point_2),
        где точки также являются кортежами координат point_x = (float, float)

        :return: кортеж двух точек ((x1, y1), (x2, y2))
        '''
        return self.line_by_coordinates

    @property
    def line_equation_calculations(self) -> dict:
        '''
        Вычисление коэффициентов k и b инициализированной прямой, в виде уравнения y=kx+b.

        :return: 1) k и b, если разности координат точек, лежащих на прямой, отличны от нуля; 2) x, если разности
                    абсцисс точек, лежащих на прямой, равны нулю; 3) y, если разности ординат точек, лежащих на
                    прямой, равны нулю.
        '''
        __x = None
        __y = None
        __k_slo = None
        __b_slo = None

        if math.fabs(self.point2[0] - self.point1[0]) < 0.0001:
            __x = self.point1[0]
        elif math.fabs(self.point2[1] - self.point1[1]) < 0.0001:
            __y = self.point1[1]
        else:
            __k_slo = (self.point2[1] - self.point1[1]) / (self.point2[0] - self.point1[0])
            __b_slo = ((self.point1[1] * (self.point2[0] - self.point1[0])) -
                       (self.point1[0] * (self.point2[1] - self.point1[1]))) / (self.point2[0] - self.point1[0])

        return {
            'x': None,
            'y': None,
            'k': __k_slo,
            'b': __b_slo,
            # 'equation': f'y={__k_slo}x'
            #             f'{f"+{__b_slo}" if __b_slo > 0 else __b_slo}'
        }

    @property
    def median_perpendicular_equation_calculations(self) -> dict:
        '''
        Вычисление коэффициентов k и b срединного перпендикуляра, в виде уравнения y=kx+b, к прямой.

        :return: 1) k и b, если разности координат точек, лежащих на срединном перпендикуляре, отличны от нуля; 2) x,
                    если разности абсцисс точек, лежащих на срединном перпендикуляре, равны нулю; 3) y, если разности
                    ординат точек, лежащих на срединном перпендикуляре, равны нулю.
        '''

        __x_perpendicular = None
        __y_perpendicular = None
        __k_perpendicular = None
        __b_perpendicular = None

        if self.line['x']:
            __y_perpendicular = self.median_point[1]
        elif self.line['y']:
            __x_perpendicular = self.median_point[0]
        else:
            __k_perpendicular = - 1 / self.line['k']
            __b_perpendicular = (self.median_point[0] / self.line['k']) + self.median_point[1]

        return {
            'x': __x_perpendicular,
            'y': __y_perpendicular,
            'k': __k_perpendicular,
            'b': __b_perpendicular,
            # 'equation': f'y={__k_perpendicular}x'
            #             f'{f"+{__b_perpendicular}" if __b_perpendicular > 0 else __b_perpendicular}'
        }

    def intersection_of_lines(self, other: 'Line', median_perpendicular=False) -> tuple[float, float]:
        '''
        Вычисление точки пересечения двух прямых.

        :param other: другая прямая.
        :param median_perpendicular: флаг, предназначенный для переключения между вычислениями точки пересечения прямых
                                     или их срединных перпендикуляров.
        :return: точка пересечения.
        '''

        if median_perpendicular:
            mp1 = self.median_perpendicular
            mp2 = other.median_perpendicular
        else:
            mp1 = self.line
            mp2 = other.line

        if mp1['x'] and mp2['y']:
            x_result = mp1['x']
            y_result = mp2['y']
        elif mp1['y'] and mp2['x']:
            x_result = mp2['x']
            y_result = mp1['y']
        elif mp1['x']:
            x_result = mp1['x']
            y_result = mp2['k'] * x_result + mp2['b']
        elif mp1['y']:
            y_result = mp1['y']
            x_result = (y_result - mp2['k']) / mp2['b']
        elif mp2['x']:
            x_result = mp2['x']
            y_result = mp1['k'] * x_result + mp1['b']
        elif mp2['y']:
            y_result = mp2['y']
            x_result = (y_result - mp1['k']) / mp1['b']
        elif mp1['k'] and mp2['k']:
            x_result = (mp1['b'] - mp2['b']) / (mp2['k'] - mp1['k'])
            y_result = mp1['k'] * x_result + mp1['b']
        else:
            return -.0, -.0

        return x_result, y_result
