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

    @property
    def x(self) -> float:
        """
        Получение абсциссы (координата X) точки.

        :return: Значение абсциссы -> (координата X).
        """

        return self._x

    @property
    def y(self) -> float:
        """
        Получение ординаты (координата Y) точки.

        :return: Значение ординаты -> (координата Y).
        """

        return self._y

    @property
    def point(self) -> tuple[float, float]:
        """
        Получение координат (координата X, координата Y) точки.

        :return: Координаты точки -> (координата X, координата Y).
        """

        return self._point

    def distance_between_points(self, other: 'Point') -> float:
        """
        Получение расстояния между точками.

        :param other: Другая точка (объект класса Point), расстояние до которой нужно вычислить.
        :return: Расстояние между точками.
        """

        return pow(pow(other.x - self.x, 2) + pow(other.y - self.y, 2), .5)

    def is_equal(self, other: 'Point') -> bool:
        if self.distance_between_points(other) < 3:
            return True
        return False

    def is_exist(self, points: list[tuple[float, float]]) -> bool:
        for item in points:
            other = Point(item[0], item[1])

            if self.distance_between_points(other) < 1:
                return True

        return False

    def is_on_line(self, line: 'Line') -> bool:
        equation = line.line_equation
        if 'k' in equation.keys():
            return True if math.fabs(equation['k'] * self._x + equation['b'] - self._y) < 2 else False
        if 'x' in equation.keys():
            return True if math.fabs(equation['x'] - self._x) < 2 else False
        if 'y' in equation.keys():
            return True if math.fabs(equation['y'] - self._y) < 2 else False


class Line(Geometry):
    """
    Класс "Прямая".
    """

    def __init__(self, point1: Point, point2: Point):
        self._point1 = point1
        self._point2 = point2

        self._line = (self._point1, self._point2)

        self._median_point = self.median_point_calculations
        self._line_equation = self.line_equation_calculations
        self._median_perpendicular_equation = self.median_perpendicular_equation_calculations

        self._color = (0, 0, 255)

    def __str__(self):
        """
        Выдача человекочитаемой информации о прямой и срединном перпендикуляре к ней.

        :return: 3 строки с информацией:
                    1) точки, через которые проходит прямая;
                    2) уравнение прямой;
                    3) уравнение срединного перпендикуляра.
        """

        result = [f'Прямая проходит через две точки --> P1{self._point1.point}, P2{self._point2.point}\n']

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

        return f'{result[0]}' \
               f'Уравнение прямой --> {result[1]["equation"]}\n' \
               f'Уравнение срединного перпендикуляра к прямой --> {result[2]["equation"]}'

    @property
    def point1(self) -> Point:
        """
        Возвращает кортеж координат первой точки отрезка (координата X, координата Y).
        """

        return self._point1

    @property
    def point2(self) -> Point:
        """
        Возвращает кортеж координат второй точки отрезка (координата X, координата Y).
        """

        return self._point2

    @property
    def line(self) -> tuple[Point, Point]:
        """
        Возвращает кортеж двух точек (point1, point2), где точки также являются кортежами координат.
        """

        return self._line

    @property
    def median_point(self) -> Point:
        """
        Возвращает кортеж координат середины отрезка (координата X, координата Y).
        """

        return self._median_point

    @property
    def line_equation(self):
        """
        Возвращает уравнение прямой.
        """

        return self._line_equation

    @property
    def median_perpendicular_equation(self):
        """
        Возвращает уравнение срединного перпендикуляра к прямой.
        """

        return self._median_perpendicular_equation

    @property
    def color(self) -> tuple[float, float, float]:
        """
        Возвращает цвет.
        :return:
        """
        return self._color

    @property
    def median_point_calculations(self) -> Point:
        """
        Вычисление координат середины отрезка.

        :return: Объект класса Point - середина отрезка.
        """
        return Point((self._point1.x + self._point2.x) / 2, (self._point1.y + self._point2.y) / 2)

    @property
    def line_equation_calculations(self) -> dict:
        """
        Вычисление коэффициентов прямой.

        :return: Коэффициенты прямой k и b или уравнение вида x=Const/y=Const.
        """

        if math.fabs(self._point2.x - self._point1.x) < 1:
            return {'x': self._point1.x}
        elif math.fabs(self._point2.y - self._point1.y) < 1:
            return {'y': self._point1.y}
        else:
            k = (self._point2.y - self._point1.y) / (self._point2.x - self._point1.x)
            b = ((self._point1.y * (self._point2.x - self._point1.x)) - (
                        self._point1.x * (self._point2.y - self._point1.y))) / (
                        self._point2.x - self._point1.x)

            return {'k': k, 'b': b}

    @property
    def median_perpendicular_equation_calculations(self) -> dict:
        """
        Вычисление коэффициентов срединного перпендикуляра к прямой.

        :return: Коэффициенты срединного перпендикуляра k и b или уравнение вида x=Const/y=Const.
        """

        if 'x' in self._line_equation.keys():
            return {'y': self._median_point.y}
        elif 'y' in self._line_equation.keys():
            return {'x': self._median_point.x}
        else:
            k = - 1 / self._line_equation['k']
            b = (self._median_point.x / self._line_equation['k']) + self._median_point.y

            return {'k': k, 'b': b}

    def intersection(self, other: 'Line', median_perpendicular=False) -> tuple[float, float] | None:
        """
        Определение пересечения двух прямых.

        :param other: Другая прямая.
        :param median_perpendicular: Флаг, определяющий уравнение самой прямой или ее срединного перпендикуляра.
        :return: Точку пересечения, в противном случае -> None.
        """

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

    def is_perpendicular(self, other: 'Line') -> bool:
        if (math.pi / 2) - 0.1 < self.angle_between_lines(other) < (math.pi / 2) + 0.1:
            return True
        return False

    def angle_between_lines(self, other: 'Line'):
        vec1 = (self._point2.x - self._point1.x, self._point2.y - self._point1.y)
        vec2 = (other.point2.x - other.point1.x, other.point2.y - other.point1.y)

        numerator = vec1[0] * vec2[0] + vec1[1] * vec2[1]
        denominator = pow(pow(vec1[0], 2) + pow(vec1[1], 2), 0.5) * pow(pow(vec2[0], 2) + pow(vec2[1], 2), 0.5)

        return math.fabs(math.acos(numerator / denominator))

    def are_equations_the_same(self, other: 'Line') -> bool:
        """
        Проверка, что уравнения двух прямых совпадают (с погрешностью).

        :param other: Объект второй прямой.
        :return: True, если уравнения совпадают, False - в противном случае.
        """
        if 'k' in self._line_equation.keys() and 'k' in other.line_equation.keys():
            if math.fabs(self._line_equation['k'] - other.line_equation['k']) < 3 and \
                    math.fabs(self._line_equation['b'] - other.line_equation['b']) < 50:
                return True
        elif 'x' in self._line_equation.keys() and 'x' in other.line_equation.keys():
            if math.fabs(self._line_equation['x'] - other.line_equation['x']) < 3:
                return True
        elif 'y' in self._line_equation.keys() and 'y' in other.line_equation.keys():
            if math.fabs(self._line_equation['y'] - other.line_equation['y']) < 3:
                return True

        return False

    def get_points_of_new_line(self, other: 'Line', mode: int, is_the_slope_positive=False):
        if mode not in (0, 1, 2):
            return

        line = None

        if mode == 0:
            max_x = max(self._point1.x, self._point2.x, other.point1.x, other.point2.x)
            max_y = max(self._point1.y, self._point2.y, other.point1.y, other.point2.y)
            min_x = min(self._point1.x, self._point2.x, other.point1.x, other.point2.x)
            min_y = min(self._point1.y, self._point2.y, other.point1.y, other.point2.y)

            if is_the_slope_positive:
                line = Line(Point(min_x, min_y), Point(max_x, max_y))
            else:
                line = Line(Point(min_x, max_y), Point(max_x, min_y))
        elif mode == 1:
            max_y = max(self._point1.y, self._point2.y, other.point1.y, other.point2.y)
            min_y = min(self._point1.y, self._point2.y, other.point1.y, other.point2.y)
            x = self._point1.x
            line = Line(Point(x, min_y), Point(x, max_y))

        elif mode == 2:
            max_x = max(self._point1.x, self._point2.x, other.point1.x, other.point2.x)
            min_x = min(self._point1.x, self._point2.x, other.point1.x, other.point2.x)
            y = self._point1.y
            line = Line(Point(min_x, y), Point(max_x, y))

        return line

    def get_new_line(self, other: 'Line'):
        mode = -1

        if 'k' in self._line_equation.keys() and 'k' in other.line_equation.keys():
            mode = 0
            is_positive = True if self._line_equation['k'] > 0 else False
            return self.get_points_of_new_line(other, mode, is_positive) if mode != -1 else None
        elif 'x' in self._line_equation.keys() and 'x' in other.line_equation.keys():
            mode = 1
        elif 'y' in self._line_equation.keys() and 'y' in other.line_equation.keys():
            mode = 2

        return self.get_points_of_new_line(other, mode) if mode != -1 else None

    def is_oppositely_directed(self, other: 'Line'):
        vec1 = (self._point1.x - self._point2.x, self._point1.y - self._point2.y)
        vec2 = (other.point1.x - other.point2.x, other.point1.y - other.point2.y)

        return all(-0.8 < vec1[i] - vec2[i] < -1.2 for i in range(len(vec1)))


class StraightAngle(Geometry):
    """
    Класс "Прямой угол"
    """

    def __init__(self, line1: Line, intersection_point: Point, line2: Line):
        self._straight_angle = (line1, intersection_point, line2)
        self._is_convex = None

    def __hash__(self):
        return hash(self._straight_angle)

    def __eq__(self, other: 'StraightAngle'):
        return self._straight_angle == other.straight_angle

    @property
    def straight_angle(self):
        """

        :return:
        """
        return self._straight_angle

    @property
    def is_convex(self) -> bool:
        """

        :return:
        """
        return self._is_convex

    def is_equal(self, other: 'StraightAngle') -> bool:
        """

        :param other:
        :return:
        """
        if self._straight_angle[1].distance_between_points(other.straight_angle[1]) < 1:
            return True
        return False

    def is_exist(self, straight_angles: list['StraightAngle']) -> bool:
        for item in straight_angles:
            if self._straight_angle[1].distance_between_points(item.straight_angle[1]) < 1:
                return True

        return False

    def convex_or_concave_angle(self, train_position: tuple[float, float]):
        """

        :param train_position:
        :return:
        """
        train_position = Point(train_position[0], train_position[1])
        opposite_side = Line(self.straight_angle[0].point2, self.straight_angle[2].point1)
        middle_of_the_opposite = opposite_side.median_point

        if train_position.distance_between_points(self.straight_angle[1]) < \
                train_position.distance_between_points(middle_of_the_opposite):
            # Выпуклый угол
            self._is_convex = True
        else:
            # Вогнутый угол
            self._is_convex = False


class Circle(Geometry):
    """
    Класс "Окружность"
    """

    def __init__(self, center: Point, radius: float):
        self._circle = (center, radius)
        self._center = center
        self._radius = radius

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


class Rectangle:
    def __init__(self, rectangle: list[StraightAngle]):
        self._rectangle = rectangle
        self._square = 0

    @property
    def rectangle(self):
        return self._rectangle

    def square_calculations(self):
        pass


class Figures:
    """
    Класс "Фигуры"
    """

    def __init__(self):
        self._points = []
        self._lines = []
        self._straight_angles: list[StraightAngle] = []
        self._circles: list[tuple[tuple[float, float], float]] = []
        self._rectangles = []

        self._field_boundaries = []
        self._groups_straight_angles: dict[StraightAngle, list[StraightAngle]] = dict()
        self._is_boundaries_found = False

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

    def get_figures(self, local_points: list[tuple[float, float]], train_position: tuple[float, float]):
        return self.get_angle(local_points, train_position) or \
               self.get_rectangle() or \
               self.get_circle(local_points) or \
               self.get_line(local_points)

    def is_point_on_the_border(self, local_point: tuple[float, float]) -> bool:
        point = Point(local_point[0], local_point[1])

        if self._is_boundaries_found:
            bound = self._field_boundaries

            for i in range(len(self._field_boundaries)):
                j = i + 1 if i < 3 else 0

                if point.is_on_line(Line(Point(bound[i][0], bound[i][1]), Point(bound[j][0], bound[j][1]))):
                    return True

        if point.is_exist(self._points):
            return True

        return False

    def is_point_on_circle(self, local_point: tuple[float, float]) -> bool:
        point = Point(local_point[0], local_point[1])

        for item in self._circles:
            circle = Circle(Point(item[0][0], item[0][1]), item[1])

            if circle.center.distance_between_points(point) < circle.radius + 3:
                return True
        return False

    def get_line(self, local_points: list[tuple[float, float]]) -> bool:
        queue_of_points = [Point(item[0], item[1]) for item in local_points]
        is_line = False

        for i in range(2):
            if queue_of_points[i].is_equal(queue_of_points[i + 1]):
                return is_line

            if queue_of_points[i].distance_between_points(queue_of_points[i + 1]) > 50:
                return is_line

            if queue_of_points[i + 1].distance_between_points(queue_of_points[i + 1]) > 50:
                return is_line

            main = Line(point1=queue_of_points[i],
                        point2=queue_of_points[i + 1])

            other = Line(point1=queue_of_points[i],
                         point2=queue_of_points[i + 2])

            buffer = None
            new_line = None

            if 'k' in main.line_equation.keys() and 'k' in other.line_equation.keys():
                equation1 = main.line_equation
                equation2 = other.line_equation

                tangent = (equation2['k'] - equation1['k']) / (1 + equation2['k'] * equation1['k'])

                if math.fabs(math.atan(tangent)) < 0.01:
                    new_line = Line(Point(other.point1.x, other.point1.y),
                                    Point(other.point2.x, other.point2.y))

            elif 'x' in main.line_equation.keys() and 'x' in other.line_equation.keys():
                if math.fabs(main.line_equation['x'] - other.line_equation['x']) < 2:
                    new_line = Line(Point(other.point1.x, other.point1.y),
                                    Point(other.point2.x, other.point2.y))

            elif 'y' in main.line_equation.keys() and 'y' in other.line_equation.keys():
                if math.fabs(main.line_equation['y'] - other.line_equation['y']) < 2:
                    new_line = Line(Point(other.point1.x, other.point1.y),
                                    Point(other.point2.x, other.point2.y))

            if self._lines:
                if new_line:
                    j = 0
                    is_one_line = False

                    while j < len(self._lines):
                        current = Line(Point(self._lines[j][0][0], self._lines[j][0][1]),
                                       Point(self._lines[j][-1][0], self._lines[j][-1][1]))

                        if (new_line.point1.is_equal(current.line[0]) or new_line.point1.is_equal(current.line[1])) and \
                                (new_line.point2.is_equal(current.line[0]) or new_line.point1.is_equal(current.line[1])):
                            j += 1
                            continue

                        if new_line.are_equations_the_same(current):
                            buffer = new_line.get_new_line(current)

                        if buffer:
                            is_line = True
                            self._lines[j] = [buffer.point1.point, buffer.point2.point]
                            is_one_line = True
                            break

                        j += 1

                    if not is_one_line:
                        is_line = True
                        self._lines.append([new_line.point1.point, new_line.point2.point])
            else:
                if new_line:
                    is_line = True
                    self._lines.append([new_line.point1.point, new_line.point2.point])

        return is_line

    def get_angle(self, local_points: list[tuple[float, float]], train_position: tuple[float, float]) -> bool:
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

        main = Line(point1=queue_of_points[1],
                    point2=queue_of_points[0])

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

            first = Line(Point(main.point2.x, main.point2.y), Point(x_inter, y_inter))
            second = Line(Point(other.point2.x, other.point2.y), Point(x_inter, y_inter))

            straight_angle = StraightAngle(first, Point(x_inter, y_inter), second)
            straight_angle.convex_or_concave_angle(train_position=train_position)

            if straight_angle.is_convex:
                if not straight_angle.is_exist(self._straight_angles):
                    self._straight_angles.append(straight_angle)
                else:
                    print(f'Выпуклый прямой угол существует! {len(self._straight_angles)}')
                    is_angle = False
            else:
                if not self._is_boundaries_found and not straight_angle.is_exist(self._field_boundaries):
                    self._field_boundaries.append(straight_angle)
                    self.get_field_boundaries()
                else:
                    print(f'Вогнутый прямой угол существует! {len(self._field_boundaries)}')
                    is_angle = False

            if is_angle:
                self._lines.append([first.point1.point, first.point2.point])
                self._lines.append([second.point1.point, second.point2.point])

        print(len(self._straight_angles))
        return is_angle

    def get_circle(self, local_points: list[tuple[float, float]]) -> bool:
        queue_of_points = [Point(item[0], item[1]) for item in local_points]
        is_circle = False

        if self.is_point_on_circle(queue_of_points[0].point):
            return False

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

    def get_rectangle(self):
        is_rectangle = False

        if not self._straight_angles or len(self._straight_angles) < 2:
            return is_rectangle

        groups = self._groups_straight_angles

        for i in range(len(self._straight_angles)):
            current = self._straight_angles[i]

            for j in range(len(self._straight_angles)):
                if i == j:
                    continue

                neighbour = self._straight_angles[j]
                is_neighbour = False
                current_line = Line(
                    Point(current.straight_angle[1].x, current.straight_angle[1].y),
                    Point(neighbour.straight_angle[1].x, neighbour.straight_angle[1].y)
                )

                if current_line.is_perpendicular(neighbour.straight_angle[0]):
                    is_neighbour = True
                elif current_line.is_perpendicular(neighbour.straight_angle[2]):
                    is_neighbour = True

                if is_neighbour:
                    if current not in groups:
                        groups[current] = [neighbour]
                    elif neighbour not in groups[current]:
                        if len(groups[current]) < 2:
                            groups[current].append(neighbour)
                        if len(groups[current]) == 2:
                            other1 = groups[current][0].straight_angle
                            other2 = groups[current][1].straight_angle
                            is_find = False
                            x = None
                            y = None

                            if not is_find:
                                if other1[0].is_perpendicular(other2[0]):
                                    x, y = other1[0].intersection(other2[0])
                                    if not Point(x, y).is_equal(current.straight_angle[1]):
                                        is_find = True
                            if not is_find:
                                if other1[0].is_perpendicular(other2[2]):
                                    x, y = other1[0].intersection(other2[2])
                                    if not Point(x, y).is_equal(current.straight_angle[1]):
                                        is_find = True

                            if not is_find:
                                if other1[2].is_perpendicular(other2[0]):
                                    x, y = other1[2].intersection(other2[0])
                                    if not Point(x, y).is_equal(current.straight_angle[1]):
                                        is_find = True

                            if not is_find:
                                if other1[2].is_perpendicular(other2[2]):
                                    x, y = other1[2].intersection(other2[2])
                                    if not Point(x, y).is_equal(current.straight_angle[1]):
                                        is_find = True

                            if is_find:
                                rect = Rectangle(rectangle=[
                                    current,
                                    groups[current][0],
                                    StraightAngle(
                                        line1=Line(other1[1], Point(x, y)),
                                        intersection_point=Point(x, y),
                                        line2=Line(Point(x, y), other2[1]),
                                    ),
                                    groups[current][1]
                                ])
                                self._rectangles.append(rect)
                                self._lines.append([rect.rectangle[0].straight_angle[1].point, rect.rectangle[1].straight_angle[1].point])
                                self._lines.append([rect.rectangle[1].straight_angle[1].point, rect.rectangle[2].straight_angle[1].point])
                                self._lines.append([rect.rectangle[2].straight_angle[1].point, rect.rectangle[3].straight_angle[1].point])
                                self._lines.append([rect.rectangle[3].straight_angle[1].point, rect.rectangle[0].straight_angle[1].point])

        return is_rectangle

    def delete_points(self):
        for item in self._points:
            bound = self._field_boundaries

            for i in range(len(bound)):
                j = i + 1 if i < 3 else 0
                point = Point(item[0], item[1])
                line = Line(Point(bound[i][0], bound[i][1]), Point(bound[j][0], bound[j][1]))

                if point.is_on_line(line):
                    self._points.remove(item)

    def get_field_boundaries(self):
        if self._field_boundaries and len(self._field_boundaries) == 3:
            x = [item.straight_angle[1].x for item in self._field_boundaries]
            y = [item.straight_angle[1].y for item in self._field_boundaries]

            top_left = (min(x), max(y))
            top_right = (max(x), max(y))
            bottom_right = (max(x), min(y))
            bottom_left = (min(x), min(y))

            self._lines.append([top_left, top_right])
            self._lines.append([top_right, bottom_right])
            self._lines.append([bottom_right, bottom_left])
            self._lines.append([bottom_left, top_left])

            self._field_boundaries = [top_left, top_right, bottom_right, bottom_left]

            self._is_boundaries_found = True
