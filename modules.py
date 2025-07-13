import time
from random import randint, sample, random, choice
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
import pickle

os.system('chcp 65001 > nul')
class Matrix():
    __slots__ = ("M", "N", "t1", "t2", "T", "intervals")
    def __init__(self, M:int, N:int, t1:int, t2:int, number=255):
        self.M = M
        self.N = N
        self.t1 = t1
        self.t2 = t2
        self.T = self.__generate_elements()
        self.intervals = self.__divide_into_intervals(self.N, number)
        pass
    def __generate_elements(self):
        def add_inf(T, N:int): # добавляю в случайное место в строках бесконечность
            for row in T:
                if randint(0, 1):  # шанс изменения строки
                    indices_to_change = sample(range(N), randint(1, N-1)) # случайные индексы для вставки бесконечности
                    for idx in indices_to_change:
                        row[idx] = "inf"
            return T

        T = [[randint(self.t1, self.t2) for j in range(self.N)] for i in range(self.M)]
        # Преобразуем в список списков, так как numpy не обязателен
        has_inf = any("inf" in row for row in T)
        while not has_inf: # генерировать бесконечности в матрицу пока не будет хотя бы 1
            T = add_inf(T, self.N)
            has_inf = any("inf" in row for row in T)
        return T
    
    def __divide_into_intervals(self, N, number): #принимает конечное число интервала и количество необходимых интервалов, возвращает список кортежей интервалов
        interval_size = number // N
        intervals = []
        start = 1
        for i in range(N):
            end = start + interval_size - 1
            intervals.append((start, end))
            start = end + 1 if i < (N-1) else end
        intervals[-1] = (intervals[-1][0], number)
        return intervals


class Organism(Matrix):
    __slots__ = ("phenotype", "genotype", "p_i", "adaptation")
    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], Matrix):
            # Если передан готовый объект Matrix, копируем его атрибуты
            matrix = args[0]
            super().__init__(matrix.M, matrix.N, matrix.t1, matrix.t2)  # вызываем родительский __init__
            self.T = matrix.T  # перезаписываем T, чтобы не генерировать заново
        else:
            # Иначе создаём новую Matrix стандартным способом
            M, N, t1, t2 = args
            super().__init__(M, N, t1, t2)
        
        p, g, p_indices = self.__assign_genotype()
        self.phenotype, self.genotype = p, g
        self.p_i = p_indices
        self.adaptation = None
    def __eq__(self, other):
        if isinstance(other, Organism):
            return {
                self.adaptation == other.adaptation and
                self.genotype == other.genotype and
                self.phenotype == other.phenotype
            }
        return False
    def __assign_genotype(self):
        p, g = [], []
        p_indices = []
        for row in self.T:
            # Находим индексы элементов, которые не являются "inf" (бесконечностью)
            indices = [i for i, x in enumerate(row) if x != "inf"]
            
            # Случайно выбираем индекс из доступных (как минимум 1 есть по условию)
            random_ind = choice(indices)
            p_indices.append(random_ind)
            p.append(row[random_ind])
            
            # Снова выбираем случайный индекс (возможно, тот же самый)
            random_ind = choice(indices)
            g.append(randint(*self.intervals[random_ind]))
            
        return p, g, p_indices
    def printf(self):
        print(self.T)
        print(self.p_i)
        print(self.phenotype)
        print(self.genotype)
        print(self.adaptation)

def Sort(method: int, matrix):
    match method:
        case 1:  # Сортировка по весу заданий по убыванию, игнорируя бесконечности
            def sum_ignoring_inf(row):
                return sum(x for x in row if x != "inf")
            # Сортируем матрицу по убыванию суммы элементов строки (игнорируя "inf")
            sorted_matrix = sorted(matrix, key=sum_ignoring_inf, reverse=True)
            return sorted_matrix
            
        case 2:  # Сначала строки с бесконечностями (сортированные по весу), затем остальные
            rows_with_inf = [row for row in matrix if "inf" in row]
            rows_without_inf = [row for row in matrix if "inf" not in row]
            # Сортируем обе группы по убыванию суммы (игнорируя "inf")
            first = Sort(1, rows_with_inf)
            second = Sort(1, rows_without_inf)
            return first + second
            
        case 3:  # Сортировка по количеству бесконечностей, затем по убыванию весов
            rows_with_inf = [row for row in matrix if "inf" in row]
            rows_without_inf = [row for row in matrix if "inf" not in row]
            # Сортируем строки с бесконечностями:
            # 1. По количеству "inf" (по убыванию)
            # 2. По минимальному значению в строке (по возрастанию, поэтому reverse=True)
            def sort_key(row):
                inf_count = row.count("inf")
                min_val = min(x for x in row if x != "inf")
                return (-inf_count, -min_val)
            
            first = sorted(rows_with_inf, key=sort_key)
            second = Sort(1, rows_without_inf)
            
            return first + second
            
        case _:
            return matrix

class Plotnikov_Zverev():
    __slots__ = ("method", "matrix", "T", "N", "M")
    def __init__(self, method: int, matrix):
        self.method = method
        self.matrix = Sort(self.method, matrix.T)
        matrix.T = self.matrix
        self.N, self.M = matrix.N, matrix.M
        
    def run(self):
        load = [0] * self.N  # Инициализация нагрузки нулями
        path = []
        tmp = [row.copy() for row in self.matrix]  # Копирование исходной матрицы

        for row_idx in range(self.M):
            # Добавляем текущую нагрузку к строке матрицы
            current_row = []
            for i, val in enumerate(self.matrix[row_idx]):
                if val == "inf":
                    current_row.append("inf")  # Бесконечности оставляем без изменений
                else:
                    current_row.append(val + load[i])  # Числа складываем с нагрузкой
            
            # Находим минимальный элемент (игнорируя "inf")
            min_val = float('inf')
            min_idx = -1
            for i, val in enumerate(current_row):
                if val != "inf" and val < min_val:
                    min_val = val
                    min_idx = i
            
            # Обновляем нагрузку и путь (гарантированно найдётся минимум)
            load[min_idx] = min_val
            path.append((tmp[row_idx][min_idx], min_idx))
        
        return path

class Goldberg():
    __slots__ = ("Z", "K", "PK", "PM", "M", "N", "t1", "t2", "elite", "elite_type", "elite_number", "system", "generations", "best_adaptation", "save_matrices", "best_path")
    def __init__(self, Z:int, K:int, PK:float, PM:float, M:int, N:int, t1:int, t2:int, elite=False, elite_type=2, elite_number=1, save_matrices=False):
        self.Z = Z #количество повторов
        self.K = K #количество особей
        self.PK = PK
        self.PM = PM
        self.elite = elite
        self.elite_type = elite_type
        self.elite_number = elite_number
        self.system = Matrix(M, N, t1, t2)
        self.generations = {}
        self.best_adaptation = None
        self.save_matrices = save_matrices
        self.best_path = None
    def __calculate_adaptation(self, organism):
            matrix = [[] for _ in range(organism.N)]
            #print(organism.phenotype)
            for i in range(len(organism.phenotype)):
                for i_nter in range(len(self.system.intervals)):
                    start, end = self.system.intervals[i_nter]
                    if start <= organism.genotype[i] <= end: #если число принадлежит интервалу
                        matrix[i_nter].append(organism.phenotype[i]) #добавить в столбец(строку) соответствующей номеру данного интервала
                        break

            '''print("=====================================")
            for i in matrix: print(i)
            print("=====================================")'''
            sums = [sum(row) for row in matrix]
            #print(sums, max(sums))
            organism.adaptation = max(sums)
            #print(organism.adaptation)
            return max(sums) #вернуть приспособляемость

    def __inverting_bit(self, num, n):
        binary_num = '{:08b}'.format(num)
        inverted_bit = '0' if binary_num[n] == '1' else '1'
        new_binary_num = binary_num[:n] + inverted_bit + binary_num[n+1:]
        inverted_int = int(new_binary_num, 2)
        return inverted_int
    
    def __crossover(self, parent1, parent2, split):
        
        child_gen = parent1.genotype[0:split] + parent2.genotype[split:]
        child_phen = parent1.phenotype[0:split] + parent2.phenotype[split:]
        child_phen_rows = parent1.T[0:split] + parent2.T[split:]
        child_phen_i = parent1.p_i[0:split] + parent2.p_i[split:]
        r = random()
        if r > self.PM and self.PM != 1 or self.PM==0:
            #print(f"вероятность оператора мутации = {r} {self.PM}, мутация не произошла")
            child = Organism(self.system)
            child.genotype = child_gen
            child.phenotype = child_phen
            child.p_i = child_phen_i
            child.T = child_phen_rows
            self.__calculate_adaptation(child)
            return child
        else:
            #print(f"вероятность оператора мутации = {r} {self.PM}, осуществление процесса мутации...")
            child = Organism(self.system)
            child.genotype = child_gen
            child.phenotype = child_phen
            child.p_i = child_phen_i
            child.T = child_phen_rows
           
            rm = 0
            for i in range(3):
                #print(f"попытка мутации {i}")
                rm = self.__mutate(child)
                if rm == 1:
                    #print("УСПЕХ")
                    break
                #print("НЕУДАЧА")
            self.__calculate_adaptation(child)
            return child

    def __mutate(self, child):
        elemind = randint(0, len(child.genotype)-1)
        p = randint(0, 7)  # всего восемь бит
        tmp = child.genotype.copy()
        mutated = self.__inverting_bit(tmp[elemind], p)
        
        for i_nter in range(len(self.system.intervals)):
            # Проверяем что значение в допустимом диапазоне и не бесконечность
            if (mutated in range(*self.system.intervals[i_nter])) and child.T[elemind][i_nter] != "inf":
                child.genotype[elemind] = mutated
                return 1  # Успешная мутация
        
        return 0  # Мутация невозможна

    def __create_elite(self):
        tmp = Organism(self.system)
        path = Plotnikov_Zverev(self.elite_type, tmp).run()
        p, g = [], []
        for i in path:
            p.append(i[0])
            g.append(randint(*(tmp.intervals[i[1]])))
        
        tmp.phenotype, tmp.genotype = p, g

        return tmp

    def __create_first_generation(self):
        gen = []
        if self.elite:
            elite = self.__create_elite()
            self.__calculate_adaptation(elite)
            for _ in range(self.elite_number):
                gen.append(elite)
            #print("СОЗДАНА ЭЛИТА")
            #elite.printf()

        while len(gen) != self.K:
            org = Organism(self.system)
            self.__calculate_adaptation(org)
            gen.append(org)
        #print(self.K)
        '''for i in gen:
            print("Фенотип: ", i.phenotype)
            print("Генотип: ",i.genotype)
            print("Адаптация: ",i.adaptation)'''
        self.generations = {0: gen}
        #print(self.generations)
        return gen
    
    def print_all_gens(self, name):
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                table {
                    border-collapse: collapse;
                    width: 100%;
                }
                th, td {
                    border: 1px solid #ddd;
                    padding: 20px;
                    text-align: left;
                    vertical-align: top;
                }
                th {
                    background-color: #f2f2f2;
                }
            </style>
        </head>
        <body>
            <table>
                <tr>
                    <th>Поколение</th>
                    <th>Фенотип</th>
                    <th>Генотип</th>
                    <th>Адаптация</th>
                </tr>
        """

        for gen_num, objects in sorted(self.generations.items()):
            # Найти минимальную адаптацию в этом поколении
            best_adaptation = min(obj.adaptation for obj in objects)
            
            phenotypes = []
            genotypes = []
            adaptations = []
            
            for obj in objects:
                # Проверяем, является ли текущая особь лучшей
                is_best = obj.adaptation == best_adaptation
                
                # Формируем строки фенотипа, генотипа и адаптации
                phenotype_str = str(obj.phenotype)
                genotype_str = str(obj.genotype)
                adaptation_str = str(obj.adaptation)
                
                if is_best:
                    phenotype_str = f"<b>{phenotype_str}</b>"
                    genotype_str = f"<b>{genotype_str}</b>"
                    adaptation_str = f"<b>{adaptation_str}</b>"
                
                phenotypes.append(phenotype_str)
                genotypes.append(genotype_str)
                adaptations.append(adaptation_str)
            
            # Собираем все особи поколения в одну строку таблицы
            html += f"""
                <tr>
                    <td>{gen_num}</td>
                    <td>{'<br>'.join(phenotypes)}</td>
                    <td>{'<br>'.join(genotypes)}</td>
                    <td>{'<br>'.join(adaptations)}</td>
                </tr>
            """
        
        html += """
            </table>
        </body>
        </html>
        """
        
        with open(f"{name}.html", "w", encoding="utf-8") as f:
            f.write(html)

    def run(self):
        gen = self.__create_first_generation()
        BEST = min(gen, key=lambda specie: specie.adaptation)
        COUNTER = 1
        dictid = 0
        while COUNTER < self.Z:
            nextGen = []

            for i in range(len(gen)):
                r = random()
                if r > self.PK and self.PK != 1:
                    nextGen.append(gen[i])

                else:
                    partner_id = i
                    while partner_id == i:
                        partner_id = randint(0, (len(gen)-1))
                    
                    split = randint(0, self.system.M-1)
                    child1 = self.__crossover(gen[i], gen[partner_id], split)
                    child2 = self.__crossover(gen[partner_id], gen[i], split)
                    contest = [gen[i], child1, child2]
                    winner = min(contest, key=lambda specie: specie.adaptation)
                    nextGen.append(winner)

            newBest = min(nextGen, key=lambda specie: specie.adaptation)
            if newBest.adaptation == BEST.adaptation and newBest.genotype == BEST.genotype and newBest.phenotype == BEST.phenotype:
                BEST = BEST
                COUNTER += 1
            elif newBest.adaptation < BEST.adaptation:
                BEST = newBest
                COUNTER = 1
            else:
                BEST = BEST
                COUNTER += 1

            dictid += 1
            self.generations[dictid] = nextGen
            gen = nextGen
        self.best_adaptation = BEST.adaptation
        #self.best_path = BEST

        matrix = [[] for _ in range(BEST.N)]
        #print(BEST.phenotype)
        for i in range(len(BEST.phenotype)):
            for i_nter in range(len(self.system.intervals)):
                start, end = self.system.intervals[i_nter]
                if start <= BEST.genotype[i] <= end: #если число принадлежит интервалу
                    matrix[i_nter].append(BEST.phenotype[i]) #добавить в столбец(строку) соответствующей номеру данного интервала
                    break
        self.best_path = matrix


# Генерация и сохранение матриц
def generate_and_save_matrices(num_matrices, M, N, t1, t2, filename="saved_matrices.pkl"):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            matrices = pickle.load(f)
        #print(f"Загружено {len(matrices)} матриц из файла")
    else:
        matrices = [Matrix(M=M, N=N, t1=t1, t2=t2) for _ in range(num_matrices)]
        with open(filename, 'wb') as f:
            pickle.dump(matrices, f)
        
        # Сохраняем Matrix.T в текстовый файл для отладки
        debug_filename = "matrix_debug.txt"
        with open(debug_filename, 'w') as debug_file:
            for i, matrix in enumerate(matrices):
                debug_file.write(f"Matrix {i} T:\n{matrix.T}\n\n")
        
        print(f"Сгенерировано и сохранено {len(matrices)} матриц")
        print(f"Для отладки матрицы сохранены в {debug_filename}")
    return matrices

# Модифицированная функция для запуска Goldberg с указанной матрицей
def run_goldberg_single(task_id, Z, K, PK, PM, M, N, t1, t2, elite, elite_type, elite_number, matrix, return_path=False):
    start_time = time.time()  # Засекаем время начала
    
    g = Goldberg(Z=Z, K=K, PK=PK, PM=PM, M=M, N=N, t1=t1, t2=t2,
                elite=elite, elite_type=elite_type, elite_number=elite_number)
    
    # Используем переданную матрицу
    g.system = matrix
    #print(g.system)
    g.run()
    exec_time = time.time() - start_time  # Вычисляем время выполнения
    
    if return_path:
        return task_id, g.best_adaptation, exec_time, g.best_path
    else:
        return task_id, g.best_adaptation, exec_time



def test_parallel(num_iters, config):
    N = config["N"]
    M = config["M"]
    t1 = config["t1"]
    t2 = config["t2"]
    PK = config["PK"]
    PM = config["PM"]
    K = config["K"]
    Z = config["Z"]
    elite_number = config["elite_number"]

    matrices = generate_and_save_matrices(num_iters, M, N, t1, t2)
    if len(matrices) < num_iters:
        print(f"Ошибка: требуется {num_iters} матриц, доступно {len(matrices)}")
        return

    execution_times = {name: [] for name in ["NoElite", "EliteType1", "EliteType2", "EliteType3"]}
    results = {name: [] for name in ["NoElite", "EliteType1", "EliteType2", "EliteType3"]}
    algorithm_real_times = {}
    all_iteration_results = {name: [] for name in ["NoElite", "EliteType1", "EliteType2", "EliteType3"]}
    best_paths = {} if num_iters == 1 else None
    task_configs = [
        ("NoElite", False, None),
        ("EliteType1", True, 1),
        ("EliteType2", True, 2),
        ("EliteType3", True, 3),
    ]
    
    def make_callback(task_name, need_path):
        def callback(fut):
            try:
                if need_path:
                    name, result, exec_time, path = fut.result()
                    best_paths[name] = path
                else:
                    name, result, exec_time = fut.result()
                
                results[name].append(result)
                execution_times[name].append(exec_time)
                all_iteration_results[name].append(result)
            except Exception as e:
                print(f"Ошибка в задаче {task_name}: {e}")
        return callback

    with ProcessPoolExecutor(max_workers=4) as executor:
        for name, elite, elite_type in task_configs:
            start_time = time.time()
            
            futures = []
            for i in range(num_iters):
                fut = executor.submit(
                        run_goldberg_single,
                        name, Z, K, PK, PM, M, N, t1, t2,
                        elite, elite_type, elite_number, matrices[i],
                        return_path=(num_iters == 1)
                )
                fut.add_done_callback(make_callback(name, num_iters == 1))
                futures.append(fut)
            
            for fut in futures:
                fut.result()
            
            algorithm_real_times[name] = time.time() - start_time

    # Построение графиков
    plt.figure(figsize=(12, 6))
    
    # График средних значений приспособленности
    plt.subplot(1, 2, 1)
    methods = list(results.keys())
    avg_fitness = [sum(results[key]) / len(results[key]) for key in methods]
    plt.bar(methods, avg_fitness)
    plt.title('Средняя приспособленность по методам')
    plt.ylabel('Приспособленность')
    plt.xticks(rotation=45)
    
    # График времени выполнения
    plt.subplot(1, 2, 2)
    real_times = [algorithm_real_times[key] for key in methods]
    plt.bar(methods, real_times, color='orange')
    plt.title('Время выполнения по методам')
    plt.ylabel('Время (сек)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Сохранение графиков
    formatted = time.strftime("%d.%m.%Y", time.localtime())
    plot_filename = f"static/parallel_graph.png"
    plt.savefig(plot_filename)
    plt.close()

    # График сходимости по итерациям
    plt.figure(figsize=(10, 6))
    for name in all_iteration_results:
        plt.plot(all_iteration_results[name], label=name)
    plt.title('Сходимость по итерациям')
    plt.xlabel('Итерация')
    plt.ylabel('Приспособленность')
    plt.legend()
    convergence_filename = f"p_convergence {formatted} elites({elite_number}) Z({Z}) K({K}) M({M}) N({N}) iterations({num_iters}).png"
    plt.savefig(convergence_filename)
    plt.close()

    # Сохранение результатов в файл
    filename = f"p {formatted} elites({elite_number}) Z({Z}) K({K}) M({M}) N({N}) iterations({num_iters}).txt"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(time.strftime("%d.%m.%Y %H:%M", time.localtime()) + "\n")
        f.write(f"{'Метод':<15} | {'Ср. приспособл.':<15} | {'Время':<15} | {'Ср. время':<15}\n")


        
        for key in results:
            avg_fitness = sum(results[key]) / len(results[key])
            real_time = algorithm_real_times[key]
            av_time = real_time / num_iters
            f.write(f"{key:<15} | {avg_fitness:<15.4f} | {real_time:<15.2f} | {av_time:<15.2f}\n")
        
        if num_iters == 1 and best_paths:
            f.write("\nЛУЧШИЕ РАСПРЕДЕЛЕНИЯ:\n")
            for name, path in best_paths.items():
                f.write(f"{name}:\n")
                for row in path:
                    # Преобразуем каждое число в строку и соединяем с табуляцией
                    f.write(f"{row}\n")
                f.write("\n")  # Добавляем пустую строку между разными путями

        f.write("\n")

        f.write("ПАРАМЕТРЫ:\n")
        f.write(f"Количество процессоров N: {N}\n")
        f.write(f"Количество задач M: {M}\n")
        f.write(f"Диапазон значений времен выполнения задач: {t1}-{t2}\n")
        f.write(f"Вероятность кроссовера PK: {PK}\n")
        f.write(f"Вероятность мутации PM: {PM}\n")
        f.write(f"Количество особей в поколении K: {K}\n")
        f.write(f"Количество повторов лучшей особи Z: {Z}\n")
        f.write(f"Количество элит в начальном поколении: {elite_number}\n")
        f.write(f"Количество ИТЕРАЦИЙ для средних значений: {num_iters}\n")

def test_sequential(num_iters, config):
    N = config["N"]
    M = config["M"]
    t1 = config["t1"]
    t2 = config["t2"]
    PK = config["PK"]
    PM = config["PM"]
    K = config["K"]
    Z = config["Z"]
    elite_number = config["elite_number"]
    
    matrices = generate_and_save_matrices(num_iters, M, N, t1, t2)
    if len(matrices) < num_iters:
        print(f"Ошибка: требуется {num_iters} матриц, доступно {len(matrices)}")
        return

    execution_times = {name: [] for name in ["NoElite", "EliteType1", "EliteType2", "EliteType3"]}
    results = {name: [] for name in ["NoElite", "EliteType1", "EliteType2", "EliteType3"]}
    algorithm_real_times = {}
    all_iteration_results = {name: [] for name in ["NoElite", "EliteType1", "EliteType2", "EliteType3"]}
    best_paths = {} if num_iters == 1 else None  # Добавлено для хранения лучших путей

    task_configs = [
        ("NoElite", False, None),
        ("EliteType1", True, 1),
        ("EliteType2", True, 2),
        ("EliteType3", True, 3),
    ]
    
    for name, elite, elite_type in task_configs:
        start_time = time.time()
        
        for i in range(num_iters):
            g = Goldberg(Z=Z, K=K, PK=PK, PM=PM, M=M, N=N, t1=t1, t2=t2,
                       elite=elite, elite_type=elite_type, elite_number=elite_number)
            g.system = matrices[i]
            g.run()
            
            results[name].append(g.best_adaptation)
            all_iteration_results[name].append(g.best_adaptation)
            execution_times[name].append(time.time() - start_time)
            
            # Сохраняем лучший путь, если это единственная итерация
            if num_iters == 1:
                best_paths[name] = g.best_path
        
        algorithm_real_times[name] = time.time() - start_time

    # Построение графиков
    plt.figure(figsize=(12, 6))
    
    # График средних значений приспособленности
    plt.subplot(1, 2, 1)
    methods = list(results.keys())
    avg_fitness = [sum(results[key]) / len(results[key]) for key in methods]
    plt.bar(methods, avg_fitness)
    plt.title('Средняя приспособленность по методам')
    plt.ylabel('Приспособленность')
    plt.xticks(rotation=45)
    
    # График времени выполнения
    plt.subplot(1, 2, 2)
    real_times = [algorithm_real_times[key] for key in methods]
    plt.bar(methods, real_times, color='orange')
    plt.title('Время выполнения по методам')
    plt.ylabel('Время (сек)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Сохранение графиков
    formatted = time.strftime("%d.%m.%Y", time.localtime())
    plot_filename = f"static/sequential_graph.png"
    plt.savefig(plot_filename)
    plt.close()

    # График сходимости по итерациям
    plt.figure(figsize=(10, 6))
    for name in all_iteration_results:
        plt.plot(all_iteration_results[name], label=name)
    plt.title('Сходимость по итерациям')
    plt.xlabel('Итерация')
    plt.ylabel('Приспособленность')
    plt.legend()
    convergence_filename = f"s_convergence {formatted} elites({elite_number}) Z({Z}) K({K}) M({M}) N({N}) iterations({num_iters}).png"
    plt.savefig(convergence_filename)
    plt.close()

    # Сохранение результатов в файл
    formatted = time.strftime("%d.%m.%Y", time.localtime())
    filename = f"s {formatted} elites({elite_number}) Z({Z}) K({K}) M({M}) N({N}) iterations({num_iters}).txt"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(time.strftime("%d.%m.%Y %H:%M", time.localtime()) + "\n")
    
        f.write(f"{'Метод':<15} | {'Ср. приспособл.':<15} | {'Время':<15} | {'Ср. время':<15}\n")
        
        
        for key in results:
            avg_fitness = sum(results[key]) / len(results[key])
            real_time = algorithm_real_times[key]
            av_time = real_time / num_iters
            f.write(f"{key:<15} | {avg_fitness:<15.4f} | {real_time:<15.2f} | {av_time:<15.2f}\n")
        
        # Добавлен блок записи лучших путей для num_iters == 1
        if num_iters == 1 and best_paths:
            f.write("\nЛУЧШИЕ РАСПРЕДЕЛЕНИЯ:\n")
            for name, path in best_paths.items():
                f.write(f"{name}:\n")
                for row in path:
                    f.write(f"{row}\n")
                f.write("\n")  # Добавляем пустую строку между разными путями

        f.write("\n")

        f.write("ПАРАМЕТРЫ:\n")
        f.write(f"Количество процессоров N: {N}\n")
        f.write(f"Количество задач M: {M}\n")
        f.write(f"Диапазон значений времен выполнения задач: {t1}-{t2}\n")
        f.write(f"Вероятность кроссовера PK: {PK}\n")
        f.write(f"Вероятность мутации PM: {PM}\n")
        f.write(f"Количество особей в поколении K: {K}\n")
        f.write(f"Количество повторов лучшей особи Z: {Z}\n")
        f.write(f"Количество элит в начальном поколении: {elite_number}\n")
        f.write(f"Количество ИТЕРАЦИЙ для средних значений: {num_iters}\n")