import math


class Geometry:
    def __init__(self, color: tuple[int, int, int] = (0, 0, 0)):
        self.__color = color
        self.__square = 0

    @property
    def color(self) -> tuple[int, int, int]:
        """
        Получение цвета в RGB палитре.

        :return: tuple[0-255, 0-255, 0-255].
        """
        return self.__color

    @color.setter
    def color(self, value: tuple[int, int, int]) -> None:
        """
        Установка цвета в RGB палитре.

        :param value: tuple[0-255, 0-255, 0-255].
        :return: None.
        """
        self.__color = value

    def set_color(self, color: tuple[int, int, int]):
        pass

    def square(self):
        pass


class Point(Geometry):
    """
    Класс "Точка".
    """

    def __init__(self, x: float, y: float, color: tuple[int, int, int] = (0, 0, 255)):
        super().__init__(color=color)
        self.__x = x
        self.__y = y
        self.__point = (x, y)

    def __eq__(self, other: 'Point') -> bool:
        """
        Проверка совпадения точек по расстоянию между ними с некоторой погрешностью.

        :param other: Точка (объект класса Point).
        :return: True, если расстояние между точками меньше некоторого значения, иначе - False.
        """

        return True if self.distance(other) < 3 else False

    def __add__(self, other: 'Point') -> 'Line':
        """
        Объединение двух точек (объекты класса Point) в прямую (объект класса Line). Например, line = point_1 + point_2.

        :param other: Точка (объект класса Point).
        :return: прямая (объект класса Line).
        """
        return Line(start=self, end=other)

    @property
    def x(self) -> float:
        """
        Получение абсциссы (координата X) точки.

        :return: Значение абсциссы -> (координата X).
        """

        return self.__x

    @property
    def y(self) -> float:
        """
        Получение ординаты (координата Y) точки.

        :return: Значение ординаты -> (координата Y).
        """

        return self.__y

    @property
    def point(self):
        """
        Получение точки в виде кортежа из ее абсциссы (координата X) и ординаты (координата Y).

        :return: Кортеж координат -> (координата X, координата Y).
        """
        return self.__point

    def distance(self, other: 'Point') -> float:
        """
        Получение расстояния между точками.

        :param other: Другая точка (объект класса Point), расстояние до которой нужно вычислить.
        :return: Расстояние между точками.
        """

        return pow(pow(other.x - self.__x, 2) + pow(other.y - self.__y, 2), .5)

    def is_exist(self, points: list['Point']) -> bool:
        """
        Проверка наличия точки в списке точек.

        :param points: Список объектов класса Point.
        :return: True, если точка есть в списке points, иначе - False.
        """

        for other in points:
            if self.distance(other) < 3:
                return True

        return False

    def is_on_line(self, line: 'Line') -> bool:
        """
        Проверка нахождения точки на прямой линии с некоторой погрешностью.

        :param line: Прямая (объект класса Line).
        :return: True, если точка лежит на прямой, иначе - False.
        """

        equation = line.line_equation
        if 'k' in equation.keys():
            return True if math.fabs(equation['k'] * self.__x + equation['b'] - self.__y) < 3 else False
        if 'x' in equation.keys():
            return True if math.fabs(equation['x'] - self.__x) < 2 else False
        if 'y' in equation.keys():
            return True if math.fabs(equation['y'] - self.__y) < 2 else False


class Line(Geometry):
    """
    Класс "Прямая".
    """

    def __init__(self, start: Point, end: Point, color: tuple[int, int, int] = (255, 0, 0)):
        super().__init__(color=color)
        self.__start = start
        self.__end = end

        self.__line = (self.__start, self.__end)

        self.__median_point = self.median_point_calculations
        self.__line_equation = self.line_equation_calculations
        self.__median_perpendicular_equation = self.median_perpendicular_equation_calculations

        self.get_sorted_points()

    def get_sorted_points(self):
        change_it = False

        if 'x' in self.__line_equation.keys():
            if self.__start.y > self.__end.y:
                change_it = True
        elif 'y' in self.__line_equation.keys():
            if self.__start.x > self.__end.x:
                change_it = True
        else:
            if self.__start.x > self.__end.x:
                change_it = True

        if change_it:
            self.__start, self.__end = self.__end, self.__start
            self.__line = (self.__start, self.__end)

    def __str__(self):
        """
        Выдача человекочитаемой информации о прямой и срединном перпендикуляре к ней.

        :return: 3 строки с информацией:
                    1) точки, через которые проходит прямая;
                    2) уравнение прямой;
                    3) уравнение срединного перпендикуляра.
        """

        result = [f'Прямая проходит через две точки --> '
                  f'P1{(self.__start.x, self.__start.y)}, P2{(self.__end.x, self.__end.y)}\n']

        if 'x' in self.__line_equation.keys():
            result.append({'equation': f'x = {self.__line_equation["x"]}'})
        elif 'y' in self.__line_equation.keys():
            result.append({'equation': f'y = {self.__line_equation["y"]}'})
        else:
            k = self.__line_equation['k']
            b = self.__line_equation['b']
            result.append({'equation': f'y = {k} x {f"+ {b}" if b > 0 else b}'})

        if 'x' in self.__median_perpendicular_equation.keys():
            result.append({'equation': f'x = {self.__median_perpendicular_equation["x"]}'})
        elif 'y' in self.__median_perpendicular_equation.keys():
            result.append({'equation': f'y = {self.__median_perpendicular_equation["y"]}'})
        else:
            k = self.__median_perpendicular_equation['k']
            b = self.__median_perpendicular_equation['b']
            result.append({'equation': f'y = {k} x {f"+ {b}" if b > 0 else b}'})

        return f'{result[0]}' \
               f'Уравнение прямой --> {result[1]["equation"]}\n' \
               f'Уравнение срединного перпендикуляра к прямой --> {result[2]["equation"]}'

    @property
    def start(self) -> Point:
        """
        Возвращает кортеж координат первой точки отрезка (координата X, координата Y).
        """

        return self.__start

    @property
    def end(self) -> Point:
        """
        Возвращает кортеж координат второй точки отрезка (координата X, координата Y).
        """

        return self.__end

    @property
    def line(self) -> tuple[Point, Point]:
        """
        Возвращает кортеж двух точек (start, end), где точки также являются кортежами координат.
        """

        return self.__line

    @property
    def median_point(self) -> Point:
        """
        Возвращает кортеж координат середины отрезка (координата X, координата Y).
        """

        return self.__median_point

    @property
    def line_equation(self):
        """
        Возвращает уравнение прямой.
        """

        return self.__line_equation

    @property
    def median_perpendicular_equation(self):
        """
        Возвращает уравнение срединного перпендикуляра к прямой.
        """

        return self.__median_perpendicular_equation

    @property
    def median_point_calculations(self) -> Point:
        """
        Вычисление координат середины отрезка.

        :return: Объект класса Point - середина отрезка.
        """
        return Point((self.__start.x + self.__end.x) / 2, (self.__start.y + self.__end.y) / 2)

    @property
    def line_equation_calculations(self) -> dict:
        """
        Вычисление коэффициентов прямой.

        :return: Коэффициенты прямой k и b или уравнение вида x=Const/y=Const.
        """

        if math.fabs(self.__end.x - self.__start.x) < 1:
            return {'x': self.__start.x}
        elif math.fabs(self.__end.y - self.__start.y) < 1:
            return {'y': self.__start.y}
        else:
            k = (self.__end.y - self.__start.y) / (self.__end.x - self.__start.x)
            b = ((self.__start.y * (self.__end.x - self.__start.x)) - (
                    self.__start.x * (self.__end.y - self.__start.y))) / (
                        self.__end.x - self.__start.x)

            return {'k': k, 'b': b}

    @property
    def median_perpendicular_equation_calculations(self) -> dict:
        """
        Вычисление коэффициентов срединного перпендикуляра к прямой.

        :return: Коэффициенты срединного перпендикуляра k и b или уравнение вида x=Const/y=Const.
        """

        if 'x' in self.__line_equation.keys():
            return {'y': self.__median_point.y}
        elif 'y' in self.__line_equation.keys():
            return {'x': self.__median_point.x}
        else:
            k = - 1 / self.__line_equation['k']
            b = (self.__median_point.x / self.__line_equation['k']) + self.__median_point.y

            return {'k': k, 'b': b}

    def get_intersection(self, other: 'Line', median_perpendicular=False) -> Point | None:
        """
        Определение точки пересечения двух прямых.

        :param other: Другая прямая (объект класса Line).
        :param median_perpendicular: Флаг, определяющий уравнение самой прямой или ее срединного перпендикуляра.
        :return: Точка пересечения (), в противном случае -> None.
        """

        if median_perpendicular:
            equation1 = self.__median_perpendicular_equation
            equation2 = other.__median_perpendicular_equation
        else:
            equation1 = self.__line_equation
            equation2 = other.__line_equation

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

        return Point(x_result, y_result) if x_result and y_result else None

    def is_perpendicular(self, other: 'Line') -> bool:
        if (math.pi / 2) - 0.1 < self.angle_between_lines(other) < (math.pi / 2) + 0.1:
            return True
        return False

    def angle_between_lines(self, other: 'Line'):
        vec1 = (self.__end.x - self.__start.x, self.__end.y - self.__start.y)
        vec2 = (other.end.x - other.start.x, other.end.y - other.start.y)

        numerator = vec1[0] * vec2[0] + vec1[1] * vec2[1]
        denominator = pow(pow(vec1[0], 2) + pow(vec1[1], 2), 0.5) * pow(pow(vec2[0], 2) + pow(vec2[1], 2), 0.5)

        return math.fabs(math.acos(numerator / denominator))

    def are_equations_the_same(self, other: 'Line') -> bool:
        """
        Проверка, что уравнения двух прямых совпадают (с погрешностью).

        :param other: Объект второй прямой.
        :return: True, если уравнения совпадают, False - в противном случае.
        """
        if 'k' in self.__line_equation.keys() and 'k' in other.line_equation.keys():
            if math.fabs(self.__line_equation['k'] - other.line_equation['k']) < 3 and \
                    math.fabs(self.__line_equation['b'] - other.line_equation['b']) < 50:
                return True
        elif 'x' in self.__line_equation.keys() and 'x' in other.line_equation.keys():
            if math.fabs(self.__line_equation['x'] - other.line_equation['x']) < 3:
                return True
        elif 'y' in self.__line_equation.keys() and 'y' in other.line_equation.keys():
            if math.fabs(self.__line_equation['y'] - other.line_equation['y']) < 3:
                return True

        return False

    def get_points_of_new_line(self, other: 'Line', mode: int, is_the_slope_positive=False):
        if mode not in (0, 1, 2):
            return

        line = None

        if mode == 0:
            max_x = max(self.__start.x, self.__end.x, other.start.x, other.end.x)
            max_y = max(self.__start.y, self.__end.y, other.start.y, other.end.y)
            min_x = min(self.__start.x, self.__end.x, other.start.x, other.end.x)
            min_y = min(self.__start.y, self.__end.y, other.start.y, other.end.y)

            if is_the_slope_positive:
                line = Line(Point(min_x, min_y), Point(max_x, max_y))
            else:
                line = Line(Point(min_x, max_y), Point(max_x, min_y))
        elif mode == 1:
            max_y = max(self.__start.y, self.__end.y, other.start.y, other.end.y)
            min_y = min(self.__start.y, self.__end.y, other.start.y, other.end.y)
            x = self.__start.x
            line = Line(Point(x, min_y), Point(x, max_y))

        elif mode == 2:
            max_x = max(self.__start.x, self.__end.x, other.start.x, other.end.x)
            min_x = min(self.__start.x, self.__end.x, other.start.x, other.end.x)
            y = self.__start.y
            line = Line(Point(min_x, y), Point(max_x, y))

        return line

    def get_new_line(self, other: 'Line'):
        mode = -1

        if 'k' in self.__line_equation.keys() and 'k' in other.line_equation.keys():
            mode = 0
            is_positive = True if self.__line_equation['k'] > 0 else False
            return self.get_points_of_new_line(other, mode, is_positive) if mode != -1 else None
        elif 'x' in self.__line_equation.keys() and 'x' in other.line_equation.keys():
            mode = 1
        elif 'y' in self.__line_equation.keys() and 'y' in other.line_equation.keys():
            mode = 2

        return self.get_points_of_new_line(other, mode) if mode != -1 else None

    def is_oppositely_directed(self, other: 'Line'):
        vec1 = (self.__start.x - self.__end.x, self.__start.y - self.__end.y)
        vec2 = (other.start.x - other.end.x, other.start.y - other.end.y)

        return all(-0.8 < vec1[i] - vec2[i] < -1.2 for i in range(len(vec1)))


class StraightAngle(Geometry):
    """
    Класс "Прямой угол"
    """

    def __init__(self, line1: Line, intersection: Point, line2: Line, color: tuple[int, int, int] = (255, 0, 247)):
        self.__first_line = line1
        self.__second_line = line2
        self.__intersection_point = intersection
        self.__straight_angle = (line1, intersection, line2)

        self._is_convex = None

        super().__init__(color=color)
        self.set_color(color)

    def __hash__(self):
        return hash(self.__straight_angle[0])

    def __eq__(self, other: 'StraightAngle'):
        return self.__straight_angle == other.straight_angle

    def set_color(self, value: tuple[int, int, int]):
        self.__first_line.color = value
        self.__second_line.color = value

    @property
    def first_line(self) -> Line:
        return self.__first_line

    @property
    def second_line(self) -> Line:
        return self.__second_line

    @property
    def straight_angle(self):
        """

        :return:
        """
        return self.__straight_angle

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
        if self.__straight_angle[1].distance(other.straight_angle[1]) < 1:
            return True
        return False

    def is_exist(self, straight_angles: list['StraightAngle']) -> bool:
        for item in straight_angles:
            if self.__straight_angle[1].distance(item.straight_angle[1]) < 1:
                return True

        return False

    def convex_or_concave_angle(self, train_position: Point):
        """

        :param train_position:
        :return:
        """

        opposite_side = Line(self.straight_angle[0].end, self.straight_angle[2].start)
        middle_of_the_opposite = opposite_side.median_point

        if train_position.distance(self.straight_angle[1]) < \
                train_position.distance(middle_of_the_opposite):
            # Выпуклый угол
            self._is_convex = True
        else:
            # Вогнутый угол
            self._is_convex = False


class Circle(Geometry):
    """
    Класс "Окружность"
    """

    def __init__(self, center: Point, radius: float, color: tuple[int, int, int] = (0, 0, 0)):
        self._circle = (center, radius)
        self._center = center
        self._radius = radius
        self.__square = 0

        self.calculating_the_square()
        super().__init__(color=color)
        self.set_color(color)
        print(f'Площадь окружности: {self.__square}')

    def set_color(self, color: tuple[int, int, int]):
        self.center.color = color

    def calculating_the_square(self):
        self.__square = math.pi * self._radius ** 2

    @property
    def circle(self) -> tuple[Point, float]:
        return self._circle

    @property
    def center(self) -> Point:
        return self._center

    @property
    def radius(self):
        return self._radius

    def is_equal(self, other: 'Circle') -> bool:
        if self.center.distance(other._center) < 1:
            return True
        return False

    def is_exist(self, circles: list['Circle']) -> bool:
        for circle in circles:
            other = Circle(center=Point(circle.center.x, circle.center.y), radius=circle.radius)
            if self.center.distance(other._center) < other.radius:
                return True

        return False


class Rectangle(Geometry):
    def __init__(self, rectangle: list[StraightAngle], color=(0, 0, 0), is_boundary=False):
        self.__rectangle = rectangle
        self.__square = 0
        self.__is_boundary = is_boundary

        super().__init__(color=color)
        self.set_color(color=color)
        self.calculating_the_square()

        if self.__is_boundary:
            print(f'Площадь поля: {self.__square}')
        else:
            print(f'Площадь прямоугольника: {self.__square}')

    def set_color(self, color):
        for angle in self.__rectangle:
            angle.first_line.color = color
            angle.second_line.color = color

    def calculating_the_square(self):
        a = self.__rectangle[0].straight_angle[1].distance(self.__rectangle[1].straight_angle[1])
        b = self.__rectangle[1].straight_angle[1].distance(self.__rectangle[2].straight_angle[1])
        self.__square = a * b

    @property
    def rectangle(self):
        return self.__rectangle

    def square_calculations(self):
        pass


class Figures:
    """
    Класс "Фигуры"
    """

    def __init__(self):
        self.__points: list[Point] = []
        self.__lines: list[Line] = []
        self.__straight_angles: list[StraightAngle] = []
        self.__circles: list[Circle] = []
        self.__rectangles: list[Rectangle] = []

        self.__field_boundary: list[StraightAngle] = []
        self.__groups_straight_angles: dict[StraightAngle, list[StraightAngle]] = dict()
        self.__is_boundary_found = False

    def __contains__(self, item):
        pass

    @property
    def points(self):
        return self.__points

    @property
    def lines(self):
        return self.__lines

    @property
    def straight_angles(self):
        return self.__straight_angles

    @property
    def circles(self):
        return self.__circles

    @property
    def field_boundaries(self):
        return self.__field_boundary

    def get_figures(self, local_points: list[Point], train_position: Point):
        return self.get_angle(local_points, train_position) or \
            self.get_rectangle() or \
            self.get_line(local_points) or \
            self.get_circle(local_points)

    def is_point_on_the_border(self, point: Point) -> bool:
        if self.__is_boundary_found:
            for i in range(len(self.__field_boundary)):
                if point.is_on_line(self.__field_boundary[i].straight_angle[0]):
                    return True

        if point.is_exist(self.__points):
            return True

        return False

    def is_point_on_circle(self, local_point: Point) -> bool:
        for circle in self.__circles:
            if circle.center.distance(local_point) < circle.radius + 3:
                return True
        return False

    def get_line(self, queue_of_points: list[Point]) -> bool:
        is_line = False

        for i in range(2):
            if queue_of_points[i] == queue_of_points[i + 1]:
                return is_line

            if queue_of_points[i].distance(queue_of_points[i + 1]) > 50:
                return is_line

            if queue_of_points[i + 1].distance(queue_of_points[i + 1]) > 50:
                return is_line

            main = queue_of_points[i] + queue_of_points[i + 1]

            other = queue_of_points[i] + queue_of_points[i + 2]

            buffer = None
            new = None

            if 'k' in main.line_equation.keys() and 'k' in other.line_equation.keys():
                equation1 = main.line_equation
                equation2 = other.line_equation

                tangent = (equation2['k'] - equation1['k']) / (1 + equation2['k'] * equation1['k'])

                if math.fabs(math.atan(tangent)) < 0.01:
                    new = other

            elif 'x' in main.line_equation.keys() and 'x' in other.line_equation.keys():
                if math.fabs(main.line_equation['x'] - other.line_equation['x']) < 2:
                    new = other

            elif 'y' in main.line_equation.keys() and 'y' in other.line_equation.keys():
                if math.fabs(main.line_equation['y'] - other.line_equation['y']) < 2:
                    new = other

            if self.__lines:
                if new:
                    j = 0
                    is_one_line = False

                    while j < len(self.__lines):
                        current = self.__lines[j]

                        if (new.start == current.line[0] or new.start == current.line[1]) and \
                                (new.end == current.line[0] or new.start == current.line[1]):
                            j += 1
                            continue

                        if new.are_equations_the_same(current):
                            buffer = new.get_new_line(current)

                        if buffer:
                            is_line = True
                            self.__lines[j] = buffer
                            is_one_line = True
                            break

                        j += 1

                    if not is_one_line:
                        is_line = True
                        self.__lines.append(new)
            else:
                if new:
                    is_line = True
                    self.__lines.append(new)

        return is_line

    def get_angle(self, local_points: list[Point], train_position: Point) -> bool:
        is_angle = False

        if local_points[0].is_exist([local_points[1], local_points[2], local_points[3]]):
            return is_angle

        distances = [
            local_points[i].distance(local_points[i + 1]) for i in range(len(local_points) - 1)
        ]

        for i in range(len(distances)):
            if distances[i] > 50:
                return is_angle

        main_line = local_points[1] + local_points[0]

        other_line = local_points[2] + local_points[3]

        if 'k' in main_line.line_equation.keys() and 'k' in other_line.line_equation.keys():
            equation1 = main_line.line_equation
            equation2 = other_line.line_equation

            tangent = (equation2['k'] - equation1['k']) / (1 + equation2['k'] * equation1['k'])

            if 1.5 < math.fabs(math.atan(tangent)) < 1.6:
                is_angle = True

        elif 'x' in main_line.line_equation.keys() and 'y' in other_line.line_equation.keys():
            is_angle = True

        elif 'y' in main_line.line_equation.keys() and 'x' in other_line.line_equation.keys():
            is_angle = True

        if is_angle:
            point = main_line.get_intersection(other_line)

            first = main_line.end + point
            second = other_line.end + point

            straight_angle = StraightAngle(first, point, second)
            straight_angle.convex_or_concave_angle(train_position=train_position)

            if straight_angle.is_convex:
                if not straight_angle.is_exist(self.__straight_angles):
                    self.__straight_angles.append(straight_angle)
                else:
                    print(f'Выпуклый прямой угол существует! {len(self.__straight_angles)}')
                    return False
            else:
                if not self.__is_boundary_found and not straight_angle.is_exist(self.__field_boundary):
                    self.__field_boundary.append(straight_angle)
                    self.get_field_boundaries()
                else:
                    print(f'Вогнутый прямой угол существует! {len(self.__field_boundary)}')
                    return False

            self.__lines.append(first)
            self.__lines.append(second)

        return is_angle

    def get_circle(self, local_points: list[Point]) -> bool:
        is_circle = False

        if self.is_point_on_circle(local_points[0]):
            return is_circle

        for i in range(3):
            if local_points[i].distance(local_points[i + 1]) > 50:
                return is_circle

        lines = [local_points[i] + local_points[j] for i in range(3) for j in range(i + 1, 4)]

        if 'k' in lines[0].line_equation.keys() and 'k' in lines[-1].line_equation.keys():
            equation1 = lines[0].line_equation
            equation2 = lines[-1].line_equation

            tangent = (equation2['k'] - equation1['k']) / (1 + equation2['k'] * equation1['k'])

            if 0.2 < math.fabs(math.atan(tangent)) < 1.0:
                is_circle = True

                intersection_points = [
                    lines[0].get_intersection(lines[1], median_perpendicular=True),
                    lines[1].get_intersection(lines[2], median_perpendicular=True),
                    lines[2].get_intersection(lines[0], median_perpendicular=True)
                ]

                center = Point(
                    (intersection_points[0].x + intersection_points[1].x + intersection_points[2].x) / 3,
                    (intersection_points[0].y + intersection_points[1].y + intersection_points[2].y) / 3
                )

                radii = [
                    center.distance(local_points[0]),
                    center.distance(local_points[1]),
                    center.distance(local_points[2]),
                ]

                radius = (radii[0] + radii[1] + radii[2]) / 3
                circle = Circle(center, radius)

                if not circle.is_exist(self.__circles):
                    self.__circles.append(circle)
                else:
                    print('Окружность существует!')
                    is_circle = False

        return is_circle

    def get_rectangle(self):
        is_rectangle = False

        if not self.__straight_angles or len(self.__straight_angles) < 2:
            return is_rectangle

        groups = self.__groups_straight_angles

        for i in range(len(self.__straight_angles)):
            current = self.__straight_angles[i]

            for j in range(len(self.__straight_angles)):
                if i == j:
                    continue

                neighbour = self.__straight_angles[j]
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
                            point = None

                            if not is_find:
                                if other1[0].is_perpendicular(other2[0]):
                                    point = other1[0].get_intersection(other2[0])
                                    if not point == current.straight_angle[1]:
                                        is_find = True
                            if not is_find:
                                if other1[0].is_perpendicular(other2[2]):
                                    point = other1[0].get_intersection(other2[2])
                                    if not point == current.straight_angle[1]:
                                        is_find = True

                            if not is_find:
                                if other1[2].is_perpendicular(other2[0]):
                                    point = other1[2].get_intersection(other2[0])
                                    if not point == current.straight_angle[1]:
                                        is_find = True

                            if not is_find:
                                if other1[2].is_perpendicular(other2[2]):
                                    point = other1[2].get_intersection(other2[2])
                                    if not point == current.straight_angle[1]:
                                        is_find = True

                            if is_find:
                                rect = Rectangle([
                                    StraightAngle(line1=other1[1] + current.straight_angle[1],
                                                  intersection=current.straight_angle[1],
                                                  line2=other2[1] + current.straight_angle[1]),

                                    StraightAngle(line1=current.straight_angle[1] + other1[1],
                                                  intersection=other1[1],
                                                  line2=point + other1[1]),

                                    StraightAngle(line1=other1[1] + point,
                                                  intersection=point,
                                                  line2=other2[1] + point),

                                    StraightAngle(line1=point + other2[1],
                                                  intersection=other2[1],
                                                  line2=current.straight_angle[1] + other2[1])
                                ],
                                    color=(0, 255, 0)
                                )

                                self.__rectangles.append(rect)
                                self.__lines.append(rect.rectangle[0].straight_angle[0])
                                self.__lines.append(rect.rectangle[0].straight_angle[2])
                                self.__lines.append(rect.rectangle[2].straight_angle[0])
                                self.__lines.append(rect.rectangle[2].straight_angle[2])

        return is_rectangle

    def get_field_boundaries(self):
        if self.__field_boundary and len(self.__field_boundary) == 3:
            x = [item.straight_angle[1].x for item in self.__field_boundary]
            y = [item.straight_angle[1].y for item in self.__field_boundary]

            top_left_point = Point(min(x), max(y))
            top_right_point = Point(max(x), max(y))
            bottom_right_point = Point(max(x), min(y))
            bottom_left_point = Point(min(x), min(y))

            boundary = Rectangle([
                StraightAngle(line1=bottom_left_point + top_left_point,
                              intersection=top_left_point,
                              line2=top_right_point + top_left_point),

                StraightAngle(line1=top_left_point + top_right_point,
                              intersection=top_right_point,
                              line2=bottom_right_point + top_right_point),

                StraightAngle(line1=top_right_point + bottom_right_point,
                              intersection=bottom_right_point,
                              line2=bottom_left_point + bottom_right_point),

                StraightAngle(line1=bottom_right_point + bottom_left_point,
                              intersection=bottom_left_point,
                              line2=top_left_point + bottom_left_point)
            ],
                color=(255, 0, 247),
                is_boundary=True
            )

            self.__lines.append(boundary.rectangle[0].first_line)
            self.__lines.append(boundary.rectangle[0].second_line)
            self.__lines.append(boundary.rectangle[2].first_line)
            self.__lines.append(boundary.rectangle[2].second_line)

            self.__is_boundary_found = True
