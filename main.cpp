#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <string>
#include "mpi.h"

using namespace std;

int N = 10000; // количество тел

bool ProveConv = true;  // проверка порядков сходимости
bool NeedWriteToFile = false; // если хотим записать в траектории
bool ManyBody = true; // много тел

const string input_file_name = "4body.txt";
const string gendata_file_name = "genbody.txt";
string output_file_name = "traj.txt";

const double G = 6.67 * 1e-11;
const double eps = 1e-3;
const double T = 20.0;
double tau = 0.01; // шаг для времени
const double output_tau = 0.1; // шаг для записи в файлы


struct Vect {
    double x1 = 0;
    double x2 = 0;
    double x3 = 0;
};

struct Body {
    Vect r; // координата
    Vect v; // скорость
    double m = 0; // масса
};

double max(const double a, const double b);
double maxof3(const double a, const double b, const double c);
double norm_minus(const Vect& vec1, const Vect& vec2);

void ReadFromFile(const string& filename, vector <Body>& data);
void generateBody(Body& body);
void GenDataInFile(int N, const string filename);
void WriteToFile(const string& file_name, double step, int num_body, const Vect& r);
void ClearFile(const std::string& file_name);

void Acceleration(const vector <Body>& data, int num, Vect& acc);
void Velocity(const vector <Body>& data, vector <Body>& result);
void SetAcceleration(const vector <Body>& data, vector <Body>& result, int len, int disp);
vector <Body> FindSol(const vector<Body>& b_prev, const vector<Body>& b_next, double h);
void RungeKutta2(vector<Body>& start, double tau, double T, const vector<int>& len, const vector<int>& disp, double& time);


int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    int myid, np; // номер текущего процесса, общее число всех процессов
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    //struct Vect
    int count = 3; //количество полей в структуре
    int length[] = { 1, 1, 1 }; //длина каждого поля
    MPI_Aint displ[] = { offsetof(Vect, x1), offsetof(Vect, x2), offsetof(Vect, x3) }; //смещения
    MPI_Datatype type[] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE }; // типы данных в структуре
    MPI_Datatype mpi_Vect;  // возвращаемый тип
    MPI_Type_create_struct(count, length, displ, type, &mpi_Vect); // создаёт структуру (количество блоков, длина, смещение каждого блока в байтах, тип данных, новый тип данных при возврате)
    MPI_Type_commit(&mpi_Vect); // Фиксирует тип данных.

    //struct Body
    int count_b = 3; //количество полей в структуре (масса, координаты, скорости)
    int length_b[] = { 1, 1, 1 };  //длина каждого поля
    MPI_Aint displ_b[] = { offsetof(Body, m), offsetof(Body, r), offsetof(Body, v) };
    MPI_Datatype type_b[] = { MPI_DOUBLE, mpi_Vect , mpi_Vect };
    MPI_Datatype mpi_Body;
    MPI_Type_create_struct(count_b, length_b, displ_b, type_b, &mpi_Body);
    MPI_Type_commit(&mpi_Body);

    vector <Body> data; // создали массив структур
    int size = 0;

    if (!ManyBody) {  //если для 4 тел, то считать из файла
        if (myid == 0) {
            ReadFromFile(input_file_name, data);
            size = data.size();
        }
    }
    else {
        if (myid == 0) { // если для многих, то генерировать и считать из этого файла
            GenDataInFile(N, gendata_file_name);
            ReadFromFile(gendata_file_name, data);
            size = data.size();
        }
    }

    MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD); // рассылка количества структур (тел) всем процессам
    data.resize(size);
    MPI_Bcast(data.data(), size, mpi_Body, 0, MPI_COMM_WORLD); // рассылка массива тел всем процессам

    vector <Body> data_copy(data);  // создаём копию массива структур

    vector<int> len, disp; // создаём массив длин и смещений
    len.resize(np);
    disp.resize(np);
    for (int i = 0; i < np; ++i) {
        len[i] = size / np;  // количество тел делим на количество процессов (размер частей массива)
    }
    for (int i = 0; i < size % np; ++i) {
        len[i] += 1;   // раздаем остатки
    }

    disp[0] = 0;
    for (int i = 1; i < np; ++i) {
        disp[i] = disp[i - 1] + len[i - 1];  // смещения (блоки)
    }

    double time = 0.0;

    RungeKutta2(data_copy, tau, T, len, disp, time);

    if (myid == 0) {
        cout << "np = " << np << "\n";
        cout << "time = " << time << "\n";
    }


    if (ProveConv) {
        //Checking convergence order

        output_file_name = "newtraj.txt";
        tau /= 2.0;
        vector <Body> data_copy1(data);
        RungeKutta2(data_copy1, tau, T, len, disp, time);

        double norm1 = 0.0;
        if (myid == 0) {
            for (int i = 0; i < data.size(); i++)
                norm1 = maxof3(fabs(data_copy[i].r.x1 - data_copy1[i].r.x1), fabs(data_copy[i].r.x2 - data_copy1[i].r.x2), fabs(data_copy[i].r.x3 - data_copy1[i].r.x3));
        }

        output_file_name = "newnewtraj.txt";
        tau /= 2.0;
        vector <Body> data_copy2(data);
        RungeKutta2(data_copy2, tau, T, len, disp, time);

        double norm2 = 0.0;
        if (myid == 0) {
            for (int i = 0; i < data.size(); i++)
                norm2 = maxof3(fabs(data_copy1[i].r.x1 - data_copy2[i].r.x1), fabs(data_copy1[i].r.x2 - data_copy2[i].r.x2), fabs(data_copy1[i].r.x3 - data_copy2[i].r.x3));
        }

        output_file_name = "newnewnewtraj.txt";
        tau /= 2.0;
        vector <Body> data_copy3(data);
        RungeKutta2(data_copy3, tau, T, len, disp, time);

        double norm3 = 0.0;
        if (myid == 0) {
            for (int i = 0; i < data.size(); i++)
                norm3 = maxof3(fabs(data_copy2[i].r.x1 - data_copy3[i].r.x1), fabs(data_copy2[i].r.x2 - data_copy3[i].r.x2), fabs(data_copy2[i].r.x3 - data_copy3[i].r.x3));
        }

        double p;
        if (myid == 0) {
            p = log(norm1 / norm3) / log(2.0);
            cout << endl << "----------------------------" << endl;
            cout << "p = " << p << endl;
        }


    }

    MPI_Finalize();

    return 0;
}



double max(const double a, const double b) {
    return ((a) > (b)) ? (a) : (b);
}

double maxof3(const double a, const double b, const double c) {
    double max = ((a) > (b)) ? (a) : (b);
    return ((max) > (c)) ? (max) : (c);
}

void ReadFromFile(const string& filename, vector <Body>& data) {
    ifstream out(filename);
    if (!out) {
        cerr << "File not found" << endl;
        exit(1);
    }
    out >> N;
    data.resize(N);
    for (int i = 0; i < N; i++) {
        out >> data[i].m >> data[i].r.x1 >> data[i].r.x2 >> data[i].r.x3   // считать из файла массу, три координаты и скорости для задачи о 4-х телах
            >> data[i].v.x1 >> data[i].v.x2 >> data[i].v.x3;
    }
    out.close();
}

void generateBody(Body& body) {   //генерировать данные для файла для задачи о большом количестве тел
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dist_m(1e6, 1e8);
    uniform_int_distribution <> dist_v_r(-10, 10);

    body.m = dist_m(gen);
    body.r.x1 = dist_v_r(gen);
    body.r.x2 = dist_v_r(gen);
    body.r.x3 = dist_v_r(gen);
    body.v.x1 = dist_v_r(gen);
    body.v.x2 = dist_v_r(gen);
    body.v.x3 = dist_v_r(gen);

}

void GenDataInFile(int N, const string filename) {   // создать файл и записать туда сгенерированные данные
    ofstream out(filename);
    if (!out) {
        cerr << "File not found!" << endl;
        exit(1);
    }
    out << N << "\n";
    Body body;
    for (int i = 0; i < N; i++) {
        generateBody(body);
        out << body.m \
            << " " << body.r.x1 << " " << body.r.x2 << " " << body.r.x3 \
            << " " << body.v.x1 << " " << body.v.x2 << " " << body.v.x3 << "\n";
    }
    out.close();
}

double norm_minus(const Vect& vec1, const Vect& vec2)   //норма разности векторов (корень из суммы квадратов разностей)
{
    double sum = 0;
    sum += (vec1.x1 - vec2.x1) * (vec1.x1 - vec2.x1);
    sum += (vec1.x2 - vec2.x2) * (vec1.x2 - vec2.x2);
    sum += (vec1.x3 - vec2.x3) * (vec1.x3 - vec2.x3);
    return sqrt(sum);
}

double norm2_minus(const Vect& vec1, const Vect& vec2)  //норма разности векторов 1 (сумма квадратов разностей)
{
    double sum = 0;
    sum += (vec1.x1 - vec2.x1) * (vec1.x1 - vec2.x1);
    sum += (vec1.x2 - vec2.x2) * (vec1.x2 - vec2.x2);
    sum += (vec1.x3 - vec2.x3) * (vec1.x3 - vec2.x3);
    return sum;
}

    const double eps3 = eps * eps * eps;  // для формулы


void Acceleration(const vector <Body>& data, int num, Vect& acc) // вычисление ускорений
{
    int N = data.size();  // количество тел в массиве
    Body my_body = data[num]; // тело,  для которого ищем ускорение

    acc.x1 = 0;
    acc.x2 = 0;
    acc.x3 = 0;

    double nrm2, idenom;

    for (int i = 0; i < N; ++i) // вычисление ускорения для i-го тела
    {
        nrm2 = norm2_minus(my_body.r, data[i].r);
        idenom = data[i].m / max(nrm2*sqrt(nrm2), eps3);
        acc.x1 += idenom * (my_body.r.x1 - data[i].r.x1);
        acc.x2 += idenom * (my_body.r.x2 - data[i].r.x2);
        acc.x3 += idenom * (my_body.r.x3 - data[i].r.x3);
    }
    acc.x1 *= -G;
    acc.x2 *= -G;
    acc.x3 *= -G;
}



void Velocity(const vector <Body>& data, vector <Body>& result) //скорости
{
    for (int i = 0; i < data.size(); i++) {
        result[i].r.x1 = data[i].v.x1;
        result[i].r.x2 = data[i].v.x2;
        result[i].r.x3 = data[i].v.x3;
    }
}

void SetAcceleration(const vector <Body>& data, vector <Body>& result, int len, int disp) {  // получаем ускорения
    Vect a;
    for (int i = disp; i < disp + len; i++) {
        Acceleration(data, i, a);

        result[i - disp].v.x1 = a.x1;
        result[i - disp].v.x2 = a.x2;
        result[i - disp].v.x3 = a.x3;

    }
}

vector <Body> FindSol(const vector<Body>& b_prev, const vector<Body>& b_next, double h) {
    int N = b_prev.size();
    vector <Body> my_body(N);

    for (int i = 0; i < N; ++i) {
        my_body[i].m = b_prev[i].m;

        my_body[i].r.x1 = b_prev[i].r.x1 + b_next[i].r.x1 * h;
        my_body[i].r.x2 = b_prev[i].r.x2 + b_next[i].r.x2 * h;
        my_body[i].r.x3 = b_prev[i].r.x3 + b_next[i].r.x3 * h;

        my_body[i].v.x1 = b_prev[i].v.x1 + b_next[i].v.x1 * h;
        my_body[i].v.x2 = b_prev[i].v.x2 + b_next[i].v.x2 * h;
        my_body[i].v.x3 = b_prev[i].v.x3 + b_next[i].v.x3 * h;
    }
    return my_body;
}

void WriteToFile(const string& file_name, double step, int num_body, const Vect& r) { // записать в файл : шаг и координаты
    ofstream out(to_string(num_body + 1) + file_name, std::ios::app);
    out << step << "    ";
    out << r.x1 << "    " << r.x2 << "    " << r.x3 << "    ";
    out << endl;
    out.close();
}

void ClearFile(const std::string& file_name) {   // очистить файл
    std::ofstream out(file_name, std::ios::trunc);
    out.close();
    out.clear();
}

void RungeKutta2(vector<Body>& start, double tau, double T, \
    const vector<int>& len, const vector<int>& disp, double& time) {

    int np, myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    int N = start.size();

    vector<Body> sol1_glob(N);
    vector<Body> sol2_glob(N);
    vector<Body> sol1(len[myid]);
    vector<Body> sol2(len[myid]);

    double half_tau = tau / 2;
    double t = 0.0;

    if (NeedWriteToFile) {
        if (myid == 0) {
            for (int i = 0; i < N; ++i) {
                ClearFile(to_string(i + 1) + output_file_name);
                WriteToFile(output_file_name, t, i, start[i].r);
            }
        }
    }

    //struct Vect
    int count = 3;
    int length[] = { 1, 1, 1 };
    MPI_Aint displ[] = { offsetof(Vect, x1), offsetof(Vect, x2), offsetof(Vect, x3) };
    MPI_Datatype type[] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE };
    MPI_Datatype mpi_Vect;
    MPI_Type_create_struct(count, length, displ, type, &mpi_Vect);
    MPI_Type_commit(&mpi_Vect);

    int count_temp = 1;
    int length_temp[] = { 1 };
    MPI_Aint displ_temp[] = { offsetof(Body, v) };
    MPI_Datatype type_temp[] = { mpi_Vect };
    MPI_Datatype mpi_temp;
    MPI_Type_create_struct(count_temp, length_temp, displ_temp, type_temp, &mpi_temp);
    MPI_Type_commit(&mpi_temp);

    MPI_Datatype mpi_Acc;
    MPI_Type_create_resized(mpi_temp, 0, sizeof(Body), &mpi_Acc);
    MPI_Type_commit(&mpi_Acc);

    double temp = 1.0;
    int iter = 0;

    if (myid == 0) {
        time = -MPI_Wtime();
    }

    while (iter <2){// (t < T + half_tau) {

        iter++;

        SetAcceleration(start, sol1, len[myid], disp[myid]);
        MPI_Allgatherv(sol1.data(), len[myid], mpi_Acc, sol1_glob.data(), len.data(), disp.data(), mpi_Acc, MPI_COMM_WORLD); // собирает решение ( откуда,сколько,тип, куда,сколько, смещения, тип, коммуникатор)
        Velocity(start, sol1_glob); //  считаем скорости

        SetAcceleration(FindSol(start, sol1_glob, half_tau), sol2, len[myid], disp[myid]);
        MPI_Allgatherv(sol2.data(), len[myid], mpi_Acc, sol2_glob.data(), len.data(), disp.data(), mpi_Acc, MPI_COMM_WORLD);
        Velocity(FindSol(start, sol1_glob, half_tau), sol2_glob);

        start = FindSol(start, sol2_glob, tau);

        t += tau;

        if (NeedWriteToFile) {
            if (myid == 0) {
                if (fabs(t - temp * output_tau) < 1e-7) {
                    for (int i = 0; i < N; ++i) {
                        WriteToFile(output_file_name, t, i, start[i].r);
                    }
                    temp += 1.0;
                }
            }
        }

    }
    if (myid == 0) {
        time += MPI_Wtime();
        time /= iter;
    }

}