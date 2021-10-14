#include <iostream>
#include <time.h>
#include <math.h>
#define NUMBER 8 //���� ��
#define HIDE 5 //������
#define OUTPUT 3 //�����
#define ROW 8 //���� ��
#define COL 6 //���� ��
#define SIZE ROW*COL //���� ũ��
#define ERROR 0.000001 //��������

using namespace std;
double eta = 0.8; //learning rate
double offset = 0.5;
int epoch = 0;
double tmp, sum;
double how[HIDE][OUTPUT], hiw[SIZE][HIDE]; //hidden_output_weight,hidden_input_weight
double bhow[HIDE][OUTPUT], bhiw[SIZE][HIDE]; //before_hidden_output_weight, befor_hidden_input_weight
double ho[HIDE]; //hidden_output
double ret[OUTPUT]; //result
double d[OUTPUT]; //delta
double hd[HIDE]; //hidden_delta
double hoe[OUTPUT][HIDE]; //hidden_output_error
double hie[HIDE][SIZE]; //hidden_input_error
double sigmoid(double a) {
	return 1 / (1 + exp(-a));
}

class Pat {
public:
	int input[SIZE]; //�Է� ����
	int teach[NUMBER]; //teach ��
	void Addinput(int arr[SIZE], int brr[NUMBER]) {
		for (int i = 0; i < SIZE; i++) {
			input[i] = arr[i];
		}
		for (int i = 0; i < NUMBER; i++) {
			teach[i] = brr[i];
		}
	}
	//�н�
	void Learn() {
		//step3-(1) �Է��� -> �������� ��� ��
		for (int i = 0; i < HIDE; i++) {
			sum = 0;
			for (int j = 0; j < SIZE; j++) {
				sum += input[j] * bhiw[j][i];
			}
			ho[i] = sigmoid(sum);
		}
		//step3-(2) ������ -> ������� ��°�
		for (int i = 0; i < OUTPUT; i++) {
			sum = 0;
			for (int j = 0; j < HIDE; j++) {
				sum += (ho[j] * bhow[j][i]);
			}
			ret[i] = sigmoid(sum);
		}
		//step3-(3) ����� �������(��Ÿ)
		for (int i = 0; i < OUTPUT; i++) {
			d[i] = ret[i] * (1 - ret[i]) * (teach[i] - ret[i]);
		}
		//step3-(4) ������ �������(��Ÿ)
		for (int i = 0; i < HIDE; i++) {
			sum = 0;
			for (int j = 0; j < OUTPUT; j++) {
				sum += d[j] * bhow[i][j];
			}
			hd[i] = ho[i] * (1 - ho[i]) * sum;
		}
		//step4-(1) ����ġ ����(������ -> �����)
		for (int i = 0; i < HIDE; i++) {
			for (int j = 0; j < OUTPUT; j++) {
				bhow[i][j] = bhow[i][j] + (eta * d[j] * ho[i]);
			}
		}
		//step4-(2) ����ġ ����(�Է��� -> ������)
		for (int i = 0; i < SIZE; i++) {
			for (int j = 0; j < HIDE; j++) {
				bhiw[i][j] = bhiw[i][j] + (eta * hd[j] * input[i]);
			}
		}
		//step5 ����ġ ��ȭ�� ����
		for (int i = 0; i < OUTPUT; i++) {
			for (int j = 0; j < HIDE; j++) {
				hoe[i][j] = how[i][j] - bhow[i][j];
				how[i][j] = bhow[i][j];
			}
		}
		for (int i = 0; i < HIDE; i++) {
			for (int j = 0; j < SIZE; j++) {
				hie[i][j] = hiw[i][j] - bhiw[i][j];
				hiw[i][j] = bhiw[i][j];
			}
		}
	}
};
int main() {
	srand(time(NULL));
	//�н�����
	int learn_input[NUMBER][SIZE] = {
	 { //�� ����
	 1,1,1,1,1,1, 1,1,1,1,1,1, 0,0,0,0,1,1, 0,0,0,0,1,1, 0,0,0,0,1,1, 0,0,0,0,1,1, 0,0,0,0,1,1, 0,0,0,0,1,1
	},
	 { //�� ����
	 1,1,0,0,0,0, 1,1,0,0,0,0, 1,1,0,0,0,0, 1,1,0,0,0,0, 1,1,0,0,0,0, 1,1,0,0,0,0, 1,1,1,1,1,1, 1,1,1,1,1,1
	},
	 { //�� ����
	 1,1,1,1,1,1, 1,1,1,1,1,1, 1,1,0,0,0,0, 1,1,0,0,0,0, 1,1,0,0,0,0, 1,1,0,0,0,0, 1,1,1,1,1,1, 1,1,1,1,1,1
	},
	 { //�� ����
	 1,1,1,1,1,1, 1,1,1,1,1,1, 0,0,0,0,1,1, 1,1,1,1,1,1, 1,1,1,1,1,1, 1,1,0,0,0,0, 1,1,1,1,1,1, 1,1,1,1,1,1
	},
	 { //�� ����
	 1,1,1,1,1,1, 1,1,1,1,1,1, 1,1,0,0,1,1, 1,1,0,0,1,1, 1,1,0,0,1,1, 1,1,0,0,1,1, 1,1,1,1,1,1, 1,1,1,1,1,1
	},
	 { //�� ����
	 1,1,0,0,1,1, 1,1,0,0,1,1, 1,1,1,1,1,1, 1,1,1,1,1,1, 1,1,0,0,1,1, 1,1,0,0,1,1, 1,1,1,1,1,1, 1,1,1,1,1,1
	},
	 { //�� ����
	 1,1,1,1,1,1, 1,1,1,1,1,1, 0,0,1,1,0,0, 0,1,1,1,1,0, 1,1,1,1,1,1, 1,1,0,0,1,1, 1,1,0,0,1,1, 1,1,0,0,1,1
	},
	 { //�� ����
	 0,0,1,1,0,0, 0,1,1,1,1,0, 1,1,1,1,1,1, 1,1,0,0,1,1, 1,1,0,0,1,1, 1,1,1,1,1,1, 0,1,1,1,1,0, 0,0,1,1,0,0
	}
	};
	//�����ȣ
	int learn_teach[NUMBER][3] = {
	 {0,0,0}, {0,0,1}, {0,1,0}, {0,1,1}, {1,0,0}, {1,0,1}, {1,0,1}, {1,1,1}
	};
	Pat pattern[NUMBER];
	//����ġ �ʱ�ȭ
	for (int i = 0; i < HIDE; i++) {
		for (int j = 0; j < OUTPUT; j++) {
			bhow[i][j] = (((rand() % 10) + 1) * 0.01);
		}
	}
	for (int i = 0; i < SIZE; i++) {
		for (int j = 0; j < HIDE; j++) {
			bhiw[i][j] = (((rand() % 10) + 1) * 0.01);
		}
	}
	//�н����� �Է�
	for (int i = 0; i < NUMBER; i++) {
		pattern[i].Addinput(learn_input[i], learn_teach[i]);
	}
	//���� �н�
	while (true) {
	start:
		epoch++;//�� ����Ŭ �߰�
		//////////////////////////////////////////���� �н�
		///////////////////////////////////////////////////
		for (int i = 0; i < NUMBER; i++) {
			pattern[i].Learn();
		}
		////////////////////////////////////����ġ �������� �˻�
		//////////////////////////////////////////////
		for (int x = 0; x < NUMBER; x++) {
			for (int i = 0; i < OUTPUT; i++) {
				for (int j = 0; j < HIDE; j++) {
					if (hoe[i][j] > ERROR) {
						goto start;
					}
					else {
						goto end;
					}
				}
			}
		}
	end:
		for (int x = 0; x < NUMBER; x++) {
			for (int i = 0; i < HIDE; i++) {
				for (int j = 0; j < SIZE; j++) {
					if (hie[i][j] > ERROR) {
						goto start;
					}
				}
			}
		}
		cout << "�н� �Ϸ�" << endl;
		cout << endl;
		break;
	}
	/////////////////////////////�н� ���� ���///////////////////////////////////////
	for (int i = 0; i < NUMBER; i++) {
		cout << "�н� ���� " << i << " ";
		cout << "teach : ";
		for (int x = 0; x < 3; x++) {
			cout << pattern[i].teach[x] << " ";
		}
		cout << endl;
		for (int j = 0; j < SIZE; j++) {
			if (pattern[i].input[j] == 1) {
				cout << "��";
			}
			else {
				cout << "��";
			}
			if (j % 6 == 5) {
				cout << endl;
			}
		}
		cout << endl;
	}
	cout << endl;
	cout << "epoch = " << epoch << endl;
	cout << endl;
	////////////////////////////////////�Է�//////////////////////////////////////////////
	int user[SIZE] = {};
	double uteach[NUMBER];
	double uho[HIDE], uret[OUTPUT];
	// user_hidden_output, user_ret, user_teach
	cout << "���� �Է�(��ȿ : 1, ��ĭ : 0)" << endl;
	for (int i = 0; i < SIZE; i++) {
		cin >> user[i];
	}
	//�Է����� Ȯ��
	cout << endl;
	cout << "�Է� ����" << endl;
	for (int i = 0; i < SIZE; i++) {
		if (user[i] == 1) {
			cout << "��";
		}
		else {
			cout << "��";
		}
		if (i % 6 == 5) {
			cout << endl;
		}
	}
	cout << endl;
	////////////////////////////////////��� ���////////////////////////////////////////
	//�Է����� ������ ��°� ���
	for (int i = 0; i < HIDE; i++) {
		sum = 0;
		for (int j = 0; j < SIZE; j++) {
			tmp = user[j] * bhiw[j][i];
			sum += tmp;
		}
		uho[i] = sigmoid(sum);
	}
	//�Է����� ����� ��°� ���
	for (int i = 0; i < OUTPUT; i++) {
		sum = 0;
		for (int j = 0; j < HIDE; j++) {
			tmp = uho[j] * bhow[j][i];
			sum += tmp;
		}
		uret[i] = sigmoid(sum);
	}
	//�Է����� �����ȣ & Ȱ���Լ� ����
	for (int i = 0; i < 3; i++) {
		uret[i] = fabs(uret[i]); //����
		cout << uret[i] << " "; //Ȱ��ȭ �Լ� ���� �� ��
	}
	cout << endl;
	double temp[3] = { offset,offset,offset };
	for (int i = 0; i < 3; i++) {
		if (temp[i] < uret[i]) {
			temp[i] = uret[i];
		}
	}
	for (int i = 0; i < 3; i++) {
		uteach[i] = uret[i] / temp[i];
		if (uteach[i] != 1) {
			uteach[i] = 0;
		}
	}
	//�Է������� �����ȣ ���
	cout << endl;
	cout << "�Է� ������ teach �� : ";
	for (int i = 0; i < 3; i++) {
		cout << uteach[i] << " ";
	}
	cout << endl;
	cout << endl;
	//��������� ���
	if (uteach[0] != 1 && uteach[1] != 1 && uteach[2] != 1) { //�����ϰ� ��
		for (int i = 0; i < SIZE; i++) {
			user[i] = pattern[0].input[i];
		}
	}
	else if (uteach[0] != 1 && uteach[1] != 1 && uteach[2] == 1) { //�����ϰ� ��
		for (int i = 0; i < SIZE; i++) {
			user[i] = pattern[1].input[i];
		}
	}
	else if (uteach[0] != 1 && uteach[1] == 1 && uteach[2] != 1) { //�����ϰ� ��
		for (int i = 0; i < SIZE; i++) {
			user[i] = pattern[2].input[i];
		}
	}
	else if (uteach[0] != 1 && uteach[1] == 1 && uteach[2] == 1) { //�����ϰ� ��
		for (int i = 0; i < SIZE; i++) {
			user[i] = pattern[3].input[i];
		}
	}
	else if (uteach[0] == 1 && uteach[1] != 1 && uteach[2] != 1) { //�����ϰ� ��
		for (int i = 0; i < SIZE; i++) {
			user[i] = pattern[4].input[i];
		}
	}
	else if (uteach[0] == 1 && uteach[1] != 1 && uteach[2] == 1) { //�����ϰ� ��
		for (int i = 0; i < SIZE; i++) {
			user[i] = pattern[5].input[i];
		}
	}
	else if (uteach[0] == 1 && uteach[1] == 1 && uteach[2] != 1) { //�����ϰ� ��
		for (int i = 0; i < SIZE; i++) {
			user[i] = pattern[6].input[i];
		}
	}
	else if (uteach[0] == 1 && uteach[1] == 1 && uteach[2] == 1) { //�����ϰ� ��
		for (int i = 0; i < SIZE; i++) {
			user[i] = pattern[7].input[i];
		}
	}
	////////////////////////////////////��� ���////////////////////////////////////////
	cout << "��� ����" << endl;
	for (int i = 0; i < SIZE; i++) {
		if (user[i] == 1) {
			cout << "��";
		}
		else {
			cout << "��";
		}
		if (i % 6 == 5) {
			cout << endl;
		}
	}
	cout << endl;
	return 0;
}