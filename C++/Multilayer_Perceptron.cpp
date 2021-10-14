#include <iostream>
#include <time.h>
#include <math.h>
#define NUMBER 8 //패턴 수
#define HIDE 5 //은닉층
#define OUTPUT 3 //출력층
#define ROW 8 //패턴 행
#define COL 6 //패턴 열
#define SIZE ROW*COL //패턴 크기
#define ERROR 0.000001 //오차범위

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
	int input[SIZE]; //입력 패턴
	int teach[NUMBER]; //teach 값
	void Addinput(int arr[SIZE], int brr[NUMBER]) {
		for (int i = 0; i < SIZE; i++) {
			input[i] = arr[i];
		}
		for (int i = 0; i < NUMBER; i++) {
			teach[i] = brr[i];
		}
	}
	//학습
	void Learn() {
		//step3-(1) 입력층 -> 은닉충의 출력 값
		for (int i = 0; i < HIDE; i++) {
			sum = 0;
			for (int j = 0; j < SIZE; j++) {
				sum += input[j] * bhiw[j][i];
			}
			ho[i] = sigmoid(sum);
		}
		//step3-(2) 은닉충 -> 출력층의 출력값
		for (int i = 0; i < OUTPUT; i++) {
			sum = 0;
			for (int j = 0; j < HIDE; j++) {
				sum += (ho[j] * bhow[j][i]);
			}
			ret[i] = sigmoid(sum);
		}
		//step3-(3) 출력층 오차계산(델타)
		for (int i = 0; i < OUTPUT; i++) {
			d[i] = ret[i] * (1 - ret[i]) * (teach[i] - ret[i]);
		}
		//step3-(4) 은닉충 오차계산(델타)
		for (int i = 0; i < HIDE; i++) {
			sum = 0;
			for (int j = 0; j < OUTPUT; j++) {
				sum += d[j] * bhow[i][j];
			}
			hd[i] = ho[i] * (1 - ho[i]) * sum;
		}
		//step4-(1) 가중치 수정(은닉층 -> 출력층)
		for (int i = 0; i < HIDE; i++) {
			for (int j = 0; j < OUTPUT; j++) {
				bhow[i][j] = bhow[i][j] + (eta * d[j] * ho[i]);
			}
		}
		//step4-(2) 가중치 수정(입력층 -> 은닉층)
		for (int i = 0; i < SIZE; i++) {
			for (int j = 0; j < HIDE; j++) {
				bhiw[i][j] = bhiw[i][j] + (eta * hd[j] * input[i]);
			}
		}
		//step5 가중치 변화량 저장
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
	//학습패턴
	int learn_input[NUMBER][SIZE] = {
	 { //ㄱ 패턴
	 1,1,1,1,1,1, 1,1,1,1,1,1, 0,0,0,0,1,1, 0,0,0,0,1,1, 0,0,0,0,1,1, 0,0,0,0,1,1, 0,0,0,0,1,1, 0,0,0,0,1,1
	},
	 { //ㄴ 패턴
	 1,1,0,0,0,0, 1,1,0,0,0,0, 1,1,0,0,0,0, 1,1,0,0,0,0, 1,1,0,0,0,0, 1,1,0,0,0,0, 1,1,1,1,1,1, 1,1,1,1,1,1
	},
	 { //ㄷ 패턴
	 1,1,1,1,1,1, 1,1,1,1,1,1, 1,1,0,0,0,0, 1,1,0,0,0,0, 1,1,0,0,0,0, 1,1,0,0,0,0, 1,1,1,1,1,1, 1,1,1,1,1,1
	},
	 { //ㄹ 패턴
	 1,1,1,1,1,1, 1,1,1,1,1,1, 0,0,0,0,1,1, 1,1,1,1,1,1, 1,1,1,1,1,1, 1,1,0,0,0,0, 1,1,1,1,1,1, 1,1,1,1,1,1
	},
	 { //ㅁ 패턴
	 1,1,1,1,1,1, 1,1,1,1,1,1, 1,1,0,0,1,1, 1,1,0,0,1,1, 1,1,0,0,1,1, 1,1,0,0,1,1, 1,1,1,1,1,1, 1,1,1,1,1,1
	},
	 { //ㅂ 패턴
	 1,1,0,0,1,1, 1,1,0,0,1,1, 1,1,1,1,1,1, 1,1,1,1,1,1, 1,1,0,0,1,1, 1,1,0,0,1,1, 1,1,1,1,1,1, 1,1,1,1,1,1
	},
	 { //ㅈ 패턴
	 1,1,1,1,1,1, 1,1,1,1,1,1, 0,0,1,1,0,0, 0,1,1,1,1,0, 1,1,1,1,1,1, 1,1,0,0,1,1, 1,1,0,0,1,1, 1,1,0,0,1,1
	},
	 { //ㅇ 패턴
	 0,0,1,1,0,0, 0,1,1,1,1,0, 1,1,1,1,1,1, 1,1,0,0,1,1, 1,1,0,0,1,1, 1,1,1,1,1,1, 0,1,1,1,1,0, 0,0,1,1,0,0
	}
	};
	//교사신호
	int learn_teach[NUMBER][3] = {
	 {0,0,0}, {0,0,1}, {0,1,0}, {0,1,1}, {1,0,0}, {1,0,1}, {1,0,1}, {1,1,1}
	};
	Pat pattern[NUMBER];
	//가중치 초기화
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
	//학습패턴 입력
	for (int i = 0; i < NUMBER; i++) {
		pattern[i].Addinput(learn_input[i], learn_teach[i]);
	}
	//패턴 학습
	while (true) {
	start:
		epoch++;//한 싸이클 추가
		//////////////////////////////////////////패턴 학습
		///////////////////////////////////////////////////
		for (int i = 0; i < NUMBER; i++) {
			pattern[i].Learn();
		}
		////////////////////////////////////가중치 오차범위 검사
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
		cout << "학습 완료" << endl;
		cout << endl;
		break;
	}
	/////////////////////////////학습 패턴 출력///////////////////////////////////////
	for (int i = 0; i < NUMBER; i++) {
		cout << "학습 패턴 " << i << " ";
		cout << "teach : ";
		for (int x = 0; x < 3; x++) {
			cout << pattern[i].teach[x] << " ";
		}
		cout << endl;
		for (int j = 0; j < SIZE; j++) {
			if (pattern[i].input[j] == 1) {
				cout << "■";
			}
			else {
				cout << "□";
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
	////////////////////////////////////입력//////////////////////////////////////////////
	int user[SIZE] = {};
	double uteach[NUMBER];
	double uho[HIDE], uret[OUTPUT];
	// user_hidden_output, user_ret, user_teach
	cout << "패턴 입력(유효 : 1, 빈칸 : 0)" << endl;
	for (int i = 0; i < SIZE; i++) {
		cin >> user[i];
	}
	//입력패턴 확인
	cout << endl;
	cout << "입력 패턴" << endl;
	for (int i = 0; i < SIZE; i++) {
		if (user[i] == 1) {
			cout << "■";
		}
		else {
			cout << "□";
		}
		if (i % 6 == 5) {
			cout << endl;
		}
	}
	cout << endl;
	////////////////////////////////////결과 계산////////////////////////////////////////
	//입력패턴 은닉층 출력값 계산
	for (int i = 0; i < HIDE; i++) {
		sum = 0;
		for (int j = 0; j < SIZE; j++) {
			tmp = user[j] * bhiw[j][i];
			sum += tmp;
		}
		uho[i] = sigmoid(sum);
	}
	//입력패턴 출력층 출력값 계산
	for (int i = 0; i < OUTPUT; i++) {
		sum = 0;
		for (int j = 0; j < HIDE; j++) {
			tmp = uho[j] * bhow[j][i];
			sum += tmp;
		}
		uret[i] = sigmoid(sum);
	}
	//입력패턴 교사신호 & 활성함수 적용
	for (int i = 0; i < 3; i++) {
		uret[i] = fabs(uret[i]); //절댓값
		cout << uret[i] << " "; //활성화 함수 적용 전 값
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
	//입력패턴의 교사신호 출력
	cout << endl;
	cout << "입력 패턴의 teach 값 : ";
	for (int i = 0; i < 3; i++) {
		cout << uteach[i] << " ";
	}
	cout << endl;
	cout << endl;
	//최종결과값 계산
	if (uteach[0] != 1 && uteach[1] != 1 && uteach[2] != 1) { //ㄱ패턴과 비교
		for (int i = 0; i < SIZE; i++) {
			user[i] = pattern[0].input[i];
		}
	}
	else if (uteach[0] != 1 && uteach[1] != 1 && uteach[2] == 1) { //ㄴ패턴과 비교
		for (int i = 0; i < SIZE; i++) {
			user[i] = pattern[1].input[i];
		}
	}
	else if (uteach[0] != 1 && uteach[1] == 1 && uteach[2] != 1) { //ㄷ패턴과 비교
		for (int i = 0; i < SIZE; i++) {
			user[i] = pattern[2].input[i];
		}
	}
	else if (uteach[0] != 1 && uteach[1] == 1 && uteach[2] == 1) { //ㄹ패턴과 비교
		for (int i = 0; i < SIZE; i++) {
			user[i] = pattern[3].input[i];
		}
	}
	else if (uteach[0] == 1 && uteach[1] != 1 && uteach[2] != 1) { //ㅁ패턴과 비교
		for (int i = 0; i < SIZE; i++) {
			user[i] = pattern[4].input[i];
		}
	}
	else if (uteach[0] == 1 && uteach[1] != 1 && uteach[2] == 1) { //ㅂ패턴과 비교
		for (int i = 0; i < SIZE; i++) {
			user[i] = pattern[5].input[i];
		}
	}
	else if (uteach[0] == 1 && uteach[1] == 1 && uteach[2] != 1) { //ㅈ패턴과 비교
		for (int i = 0; i < SIZE; i++) {
			user[i] = pattern[6].input[i];
		}
	}
	else if (uteach[0] == 1 && uteach[1] == 1 && uteach[2] == 1) { //ㅇ패턴과 비교
		for (int i = 0; i < SIZE; i++) {
			user[i] = pattern[7].input[i];
		}
	}
	////////////////////////////////////결과 출력////////////////////////////////////////
	cout << "결과 패턴" << endl;
	for (int i = 0; i < SIZE; i++) {
		if (user[i] == 1) {
			cout << "■";
		}
		else {
			cout << "□";
		}
		if (i % 6 == 5) {
			cout << endl;
		}
	}
	cout << endl;
	return 0;
}