import math


class Geometry:
    pass


class Point(Geometry):
    '''
    Класс "Точка".
    '''

    def __init__(self, x, y):
        self._x = x
        self._y = y
        self._point = (x, y)

    def __setattr__(self, key, value):
        '''
        Переопределение метода __setattr__ для невозможности создавать новые и изменять существующие атрибуты.

        :param key: Название атрибута.
        :param value: Значение атрибута.
        :return: None.
        '''

        if not hasattr(self, key) and key in ('_x', '_y', '_point'):
            super().__setattr__(key, value)
        else:
            raise AttributeError(f"can't set attribute '{key}'")

    @property
    def x(self) -> float:
        '''
        Получение абсциссы (координата X) точки.

        :return: Значение абсциссы -> (координата X).
        '''

        return self._x

    @property
    def y(self) -> float:
        '''
        Получение ординаты (координата Y) точки.

        :return: Значение ординаты -> (координата Y).
        '''

        return self._y

    @property
    def point(self) -> tuple[float, float]:
        '''
        Получение координат (координата X, координата Y) точки.

        :return: Координаты точки -> (координата X, координата Y).
        '''

        return self._point

    def distance_between_points(self, other: 'Point') -> float:
        '''
        Получение расстояния между точками.

        :param other: Другая точка (объект класса Point), расстояние до которой нужно вычислить.
        :return: Расстояние между точками.
        '''

        return pow(pow(other.x - self.x, 2) + pow(other.y - self.y, 2), .5)

    def is_equal(self, other: 'Point') -> bool:
        if self.distance_between_points(other) < 1:
            return True
        return False

    def is_exist(self, points: list[tuple[float, float]]) -> bool:
        for item in points:
            other = Point(item[0], item[1])

            if self.distance_between_points(other) < 1:
                return True

        return False


class Line(Geometry):
    '''
        Класс "Прямая".
    '''

    def __init__(self, point1: Point, point2: Point):
        self._point1 = point1
        self._point2 = point2

        self._line = (self._point1, self._point2)

        self._median_point = self.median_point_calculations
        self._line_equation = self.line_equation_calculations
        self._median_perpendicular_equation = self.median_perpendicular_equation_calculations

        self._color = (0, 0, 255)

    def __setattr__(self, key, value):
        '''
        Переопределение метода __setattr__ для невозможности создавать новые и изменять существующие атрибуты.

        :param key: Название атрибута.
        :param value: Значение атрибута.
        :return: None.
        '''

        if not hasattr(self, key) and key in ('_point1', '_point2', '_line', '_median_point', '_line_equation', '_median_perpendicular_equation', '_color'):
            super().__setattr__(key, value)
        else:
            raise AttributeError(f"can't set attribute '{key}'")

    def __str__(self):
        '''
        Выдача человекочитаемой информации о прямой и срединном перпендикуляре к ней.

        :return: 3 строки с информацией:
                    1) точки, через которые проходит прямая;
                    2) уравнение прямой;
                    3) уравнение срединного перпендикуляра.
        '''

        result = [f'Прямая проходит через две точки --> P1{self._point1}, P2{self._point2}\n']

        if 'x' in self._line_equation.keys():
            result.append({'equation': f'x = {self._line_equation["x"]}'})
        elif 'y' in self._line_equation.keys():
            result.append({'equation': f'y = {self._line_equation["y"]}'})
        else:
            k = self._line_equation['k']
            b = self._line_equation['b']
            result.append({'equation': f'y = {k} x {f"+ {b}" if b > 0 else b}'})

        if 'x' in self._median_perpendicular_equation.keys():
            result.append({'equation': f'x = {self._median_perpendicular_equation["x"]}'})
        elif 'y' in self._median_perpendicular_equation.keys():
            result.append({'equation': f'y = {self._median_perpendicular_equation["y"]}'})
        else:
            k = self._median_perpendicular_equation['k']
            b = self._median_perpendicular_equation['b']
            result.append({'equation': f'y = {k} x {f"+ {b}" if b > 0 else b}'})

        return f'{result[0]}'\
               f'Уравнение прямой --> {result[1]["equation"]}\n' \
               f'Уравнение срединного перпендикуляра к прямой --> {result[2]["equation"]}'

    @property
    def point1(self) -> Point:
        '''
        Возвращает кортеж координат первой точки отрезка (координата X, координата Y).
        '''

        return self._point1

    @property
    def point2(self) -> Point:
        '''
        Возвращает кортеж координат второй точки отрезка (координата X, координата Y).
        '''

        return self._point2

    @property
    def line(self) -> tuple[Point, Point]:
        '''
        Возвращает кортеж двух точек (point1, point2), где точки также являются кортежами координат.
        '''

        return self._line

    @property
    def median_point(self) -> Point:
        '''
        Возвращает кортеж координат середины отрезка (координата X, координата Y).
        '''

        return self._median_point

    @property
    def line_equation(self):
        '''
        Возвращает уравнение прямой.
        '''

        return self._line_equation

    @property
    def median_perpendicular_equation(self):
        '''
        Возвращает уравнение срединного перпендикуляра к прямой.
        '''

        return self._median_perpendicular_equation

    @property
    def color(self) -> tuple[float, float, float]:
        '''
        Возвращает цвет.
        :return:
        '''
        return self._color

    @property
    def median_point_calculations(self) -> Point:
        '''
        Вычисление координат середины отрезка.

        :return: Объект класса Point - середина отрезка.
        '''
        return Point((self._point1.x + self._point2.x) / 2, (self._point1.y + self._point2.y) / 2)

    @property
    def line_equation_calculations(self) -> dict:
        '''
        Вычисление коэффициентов прямой.

        :return: Коэффициенты прямой k и b или уравнение вида x=Const/y=Const.
        '''

        if self._point2.x - self._point1.x == 0:
            return {'x': self._point1.x}
        elif self._point2.y - self._point1.y == 0:
            return {'y': self._point1.y}
        else:
            k = (self._point2.y - self._point1.y) / (self._point2.x - self._point1.x)
            b = ((self._point1.y * (self._point2.x - self._point1.x)) - (self._point1.x * (self._point2.y - self._point1.y))) / (
                    self._point2.x - self._point1.x)

            return {'k': k, 'b': b}

    @property
    def median_perpendicular_equation_calculations(self) -> dict:
        '''
        Вычисление коэффициентов срединного перпендикуляра к прямой.

        :return: Коэффициенты срединного перпендикуляра k и b или уравнение вида x=Const/y=Const.
        '''

        if 'x' in self._line_equation.keys():
            return {'y': self._median_point.y}
        elif 'y' in self._line_equation.keys():
            return {'x': self._median_point.x}
        else:
            k = - 1 / self._line_equation['k']
            b = (self._median_point.x / self._line_equation['k']) + self._median_point.y

            return {'k': k, 'b': b}

    def intersection(self, other: 'Line', median_perpendicular=False) -> tuple[float, float] | None:
        '''
        Определение пересечения двух прямых.

        :param other: Другая прямая.
        :param median_perpendicular: Флаг, определяющий уравнение самой прямой или ее срединного перпендикуляра.
        :return: Точку пересечения, в противном случае -> None.
        '''

        if median_perpendicular:
            equation1 = self._median_perpendicular_equation
            equation2 = other._median_perpendicular_equation
        else:
            equation1 = self._line_equation
            equation2 = other._line_equation

        if 'x' in equation1.keys() and 'y' in equation2.keys():
            x_result = equation1['x']
            y_result = equation2['y']
        elif 'y' in equation1.keys() and 'x' in equation2.keys():
            x_result = equation2['x']
            y_result = equation1['y']
        elif 'x' in equation1.keys():
            x_result = equation1['x']
            y_result = equation2['k'] * x_result + equation2['b']
        elif 'y' in equation1.keys():
            y_result = equation1['y']
            x_result = (y_result - equation2['k']) / equation2['b']
        elif 'x' in equation2.keys():
            x_result = equation2['x']
            y_result = equation1['k'] * x_result + equation1['b']
        elif 'y' in equation2.keys():
            y_result = equation2['y']
            x_result = (y_result - equation1['k']) / equation1['b']
        else:
            x_result = (equation1['b'] - equation2['b']) / (equation2['k'] - equation1['k'])
            y_result = equation1['k'] * x_result + equation1['b']

        return x_result, y_result


class StraightAngle(Geometry):
    '''
    Класс "Прямой угол"
    '''
    def __init__(self, line1: Line, intersection_point: Point, line2: Line):
        self._straight_angle = (line1, intersection_point, line2)
        self._is_convex = None

    def __setattr__(self, key, value):
        '''
        Переопределение метода __setattr__ для невозможности создавать новые и изменять существующие атрибуты.

        :param key: Название атрибута.
        :param value: Значение атрибута.
        :return: None.
        '''

        if not hasattr(self, key) and key in ('_straight_angle', '_is_convex'):
            super().__setattr__(key, value)
        else:
            raise AttributeError(f"can't set attribute '{key}'")

    @property
    def straight_angle(self):
        '''

        :return:
        '''
        return self._straight_angle

    @property
    def is_convex(self) -> bool:
        '''

        :return:
        '''
        return self._is_convex

    def is_equal(self, other: 'StraightAngle') -> bool:
        '''

        :param other:
        :return:
        '''
        if self._straight_angle[1].distance_between_points(other.straight_angle[1]) < 1:
            return True
        return False

    def is_exist(self, straight_angles: list[list[tuple[float, float]]]) -> bool:
        for item in straight_angles:
            other = StraightAngle(
                line1=Line(
                    point1=Point(x=item[0][0], y=item[0][1]),
                    point2=Point(x=item[1][0], y=item[1][1])
                ),
                intersection_point=Point(
                    x=item[1][0],
                    y=item[1][1]
                ),

                line2=Line(
                    point1=Point(x=item[1][0], y=item[1][1]),
                    point2=Point(x=item[2][0], y=item[2][1])
                )
            )

            if self._straight_angle[1].distance_between_points(other.straight_angle[1]) < 1:
                return True

        return False

    def convex_or_concave_angle(self, train_position: Point, opposite_side: Line, top_of_angle: Point):
        '''

        :param train_position:
        :param opposite_side:
        :param top_of_angle:
        :return:
        '''
        middle_of_the_opposite = opposite_side.median_point

        if train_position.distance_between_points(top_of_angle) < \
                train_position.distance_between_points(middle_of_the_opposite):
            # Выпуклый угол
            self._is_convex = True
        else:
            # Вогнутый угол
            self._is_convex = False


class Circle(Geometry):
    '''
    Класс "Окружность"
    '''

    def __init__(self, center: Point, radius: float):
        self._circle = (center, radius)
        self._center = center
        self._radius = radius

    def __setattr__(self, key, value):
        '''
        Переопределение метода __setattr__ для невозможности создавать новые и изменять существующие атрибуты.

        :param key: Название атрибута.
        :param value: Значение атрибута.
        :return: None.
        '''

        if not hasattr(self, key) and key in ('_circle', '_center', '_radius'):
            super().__setattr__(key, value)
        else:
            raise AttributeError(f"can't set attribute '{key}'")

    @property
    def circle(self):
        return self._circle

    @property
    def center(self):
        return self._center

    @property
    def radius(self):
        return self._radius

    def is_equal(self, other: 'Circle'):
        if self.center.distance_between_points(other._center) < 1:
            return True
        return False

    def is_exist(self, circles: list[tuple[tuple[float, float], float]]):
        for item in circles:
            other = Circle(center=Point(item[0][0], item[0][1]), radius=item[1])
            if self.center.distance_between_points(other._center) < other.radius:
                return True

        return False


class Figures:
    '''
    Класс "Фигуры"
    '''

    def __init__(self):
        self._points = []
        self._lines = []
        self._straight_angles = []
        self._circles = []
        self._field_boundaries = []

    def __setattr__(self, key, value):
        '''
        Переопределение метода __setattr__ для невозможности создавать новые и изменять существующие атрибуты.

        :param key: Название атрибута.
        :param value: Значение атрибута.
        :return: None.
        '''

        if not hasattr(self, key) and key in ('_points', '_lines', '_straight_angles', '_circles', '_field_boundaries'):
            super().__setattr__(key, value)
        else:
            raise AttributeError(f"can't set attribute '{key}'")

    @property
    def points(self):
        return self._points

    @property
    def lines(self):
        return self._lines

    @property
    def straight_angles(self):
        return self._straight_angles

    @property
    def circles(self):
        return self._circles

    @property
    def field_boundaries(self):
        return self._field_boundaries

    def get_figures(self, local_points):
        if not self.get_line(local_points):
            if not self.get_angle(local_points):
                self.get_circle(local_points)

    def get_point(self, local_point: tuple[float, float]) -> bool:
        if Point(local_point[0], local_point[1]).is_exist(self._points):
            return True

        return False

    def get_line(self, local_points: list[tuple[float, float]]) -> bool:
        queue_of_points = [Point(item[0], item[1]) for item in local_points]

        is_line = False
        eps = 0.0009

        for i in range(2):
            if queue_of_points[i].is_equal(queue_of_points[i + 1]):
                return is_line

            if len(queue_of_points) > 1 and queue_of_points[i].distance_between_points(queue_of_points[i + 1]) > 50:
                return is_line

            if len(queue_of_points) > 2 and queue_of_points[i + 1].distance_between_points(queue_of_points[i + 1]) > 50:
                return is_line

            general_line = Line(point1=queue_of_points[i],
                                point2=queue_of_points[i + 2])

            other_line = Line(point1=queue_of_points[i],
                              point2=queue_of_points[i + 1])

            buffer = []

            if 'k' in general_line.line_equation.keys() and 'k' in other_line.line_equation.keys():
                equation1 = general_line.line_equation
                equation2 = other_line.line_equation

                tangent = (equation2['k'] - equation1['k']) / (1 + equation2['k'] * equation1['k'])

                if math.fabs(math.atan(tangent)) < 0.01:
                    if self._lines and len(self._lines[-1]) > 1 and self._lines[-1][-2] == queue_of_points[i].point and \
                            self._lines[-1][-1] == queue_of_points[i + 1].point:
                        self._lines[-1].append(queue_of_points[i + 2].point)
                    else:
                        buffer.append(queue_of_points[i].point)
                        buffer.append(queue_of_points[i + 1].point)
                        buffer.append(queue_of_points[i + 2].point)
                        self._lines.append(buffer)

                    is_line = True

            elif 'x' in general_line.line_equation.keys() and 'x' in other_line.line_equation.keys():
                if - eps < general_line.line_equation['x'] - other_line.line_equation['x'] < eps:
                    if self._lines and len(self._lines[-1]) > 1 and self._lines[-1][-2] == queue_of_points[i].point and \
                            self._lines[-1][-1] == queue_of_points[i + 1].point:
                        self._lines[-1].append(queue_of_points[i + 2].point)
                    else:
                        buffer.append(queue_of_points[i].point)
                        buffer.append(queue_of_points[i + 1].point)
                        buffer.append(queue_of_points[i + 2].point)
                        self._lines.append(buffer)

                    is_line = True

            elif 'y' in general_line.line_equation.keys() and 'y' in other_line.line_equation.keys():
                if - eps < general_line.line_equation['y'] - other_line.line_equation['y'] < eps:
                    if self._lines and len(self._lines[-1]) > 1 and self._lines[-1][-2] == queue_of_points[i].point and \
                            self._lines[-1][-1] == queue_of_points[i + 1].point:
                        self._lines[-1].append(queue_of_points[i + 2].point)
                    else:
                        buffer.append(queue_of_points[i].point)
                        buffer.append(queue_of_points[i + 1].point)
                        buffer.append(queue_of_points[i + 2].point)
                        self._lines.append(buffer)

                    is_line = True

            if self._lines and len(self._lines[-1]) > 2:
                buffer = [self._lines[-1][0], self._lines[-1][-1]]
                self._lines[-1] = buffer

        return is_line

    def get_angle(self, local_points: list[tuple[float, float]]) -> bool:
        queue_of_points = [Point(item[0], item[1]) for item in local_points]
        is_angle = False

        if queue_of_points[0].point in (queue_of_points[1].point, queue_of_points[2].point, queue_of_points[3].point):
            return is_angle

        distances = [
            queue_of_points[i].distance_between_points(queue_of_points[i + 1]) for i in range(len(queue_of_points) - 1)
        ]

        for i in range(len(distances)):
            if distances[i] > 50:
                return is_angle

        main = Line(point1=queue_of_points[0],
                    point2=queue_of_points[1])

        other = Line(point1=queue_of_points[2],
                     point2=queue_of_points[3])

        if 'k' in main.line_equation.keys() and 'k' in other.line_equation.keys():
            equation1 = main.line_equation
            equation2 = other.line_equation

            tangent = (equation2['k'] - equation1['k']) / (1 + equation2['k'] * equation1['k'])

            if 1.5 < math.fabs(math.atan(tangent)) < 1.6:
                is_angle = True

        elif 'x' in main.line_equation.keys() and 'y' in other.line_equation.keys():
            is_angle = True

        elif 'y' in main.line_equation.keys() and 'x' in other.line_equation.keys():
            is_angle = True

        if is_angle:
            x_inter, y_inter = main.intersection(other)

            # проверка угла на выпуклость/вогнутость
            # self.convex_or_concave_angle(queue_of_points, (x_inter, y_inter))
            straight_angle = StraightAngle(main, Point(x_inter, y_inter), other)
            if not straight_angle.is_exist(self._straight_angles):
                self._lines.append([queue_of_points[0].point, (x_inter, y_inter)])
                self._lines.append([(x_inter, y_inter), queue_of_points[3].point])
                self._straight_angles.append([queue_of_points[1].point, (x_inter, y_inter), queue_of_points[2].point])
            else:
                print(f'Прямой угол существует!')
                is_angle = False

        return is_angle

    def get_circle(self, local_points: list[tuple[float, float]]) -> bool:
        queue_of_points = [Point(item[0], item[1]) for item in local_points]
        is_circle = False

        for i in range(3):
            if queue_of_points[i].distance_between_points(queue_of_points[i + 1]) > 50:
                return is_circle

        lines = [
            Line(queue_of_points[i], queue_of_points[j]) for i in range(3) for j in range(i + 1, 4)
        ]

        if 'k' in lines[0].line_equation.keys() and 'k' in lines[-1].line_equation.keys():
            equation1 = lines[0].line_equation
            equation2 = lines[-1].line_equation

            tg_fi = (equation2['k'] - equation1['k']) / (1 + equation2['k'] * equation1['k'])

            if 0.1 < math.fabs(math.atan(tg_fi)) < 1:
                is_circle = True

                intersection_points = [
                    lines[0].intersection(lines[1], median_perpendicular=True),
                    lines[1].intersection(lines[2], median_perpendicular=True),
                    lines[2].intersection(lines[0], median_perpendicular=True)
                ]

                center = Point(
                    (intersection_points[0][0] + intersection_points[1][0] + intersection_points[2][0]) / 3,
                    (intersection_points[0][1] + intersection_points[1][1] + intersection_points[2][1]) / 3
                )

                radii = [
                    center.distance_between_points(queue_of_points[0]),
                    center.distance_between_points(queue_of_points[1]),
                    center.distance_between_points(queue_of_points[2]),
                ]

                radius = (radii[0] + radii[1] + radii[2]) / 3

                if not Circle(center, radius).is_exist(self._circles):
                    self._circles.append((center.point, radius))
                else:
                    print('Окружность существует!')
                    is_circle = False

        return is_circle
