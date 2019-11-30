#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <sstream> 
#include <cmath>
#include <ctime>
#include <cstdlib>

using namespace std;

int dimension;
int data_num;
vector< string > attribute;
map< string, vector< double > > all_data;
map< string, double > max_value;
map< string, double > min_value;
vector< string > dteday;
vector< int > realy_count;
vector< int > train_id;
vector< int > validation_id;

int layers;
double n;
vector< int > nodes;
map< int, vector< vector< double > > > weights;
map< int, vector< vector< double > > > weight_step;
map< int, vector< double > > bias;
map< int, vector< double > > bias_step;

template< class out_type, class in_type >  
out_type convert( const in_type & value ) {
    stringstream stream;  
    out_type result;
    
    stream << value;  
    stream >> result; 
    
    return result;
}

vector< double > NormalizeVector( vector< double > vec, string attr ) {
	double max = 0, min = 1000;
	vector< double > result;
	
	for( int i = 0; i < vec.size(); i ++ ) {  //找到最大值和最小值 
		if( max < vec[ i ] )
		    max = vec[ i ];
		if( min > vec[ i ] )
		    min = vec[ i ];
	}
	max_value[ attr ] = max;  //记录该列特征的最大值和最小值 
	min_value[ attr ] = min;
	for( int i = 0; i < vec.size(); i ++ )  //归一化 
		result.push_back( ( vec[ i ] - min ) / ( max - min ) );
	
	return result;
}

//测试集按训练集的最大值和最小值进行归一化 
double NormalizeValue( double val, string attr ) {
	return ( val - min_value[ attr ] ) / ( max_value[ attr ] - min_value[ attr ] );
}

void PartitionDataSet() {
	for( int i = 0; i < data_num; i ++ ) {
		if( i % 10 == 0 )  //将第1个、第11个、...、第10n+1个样本划分为验证集 
			validation_id.push_back( i );
		else  //剩余样本划分为训练集 
		    train_id.push_back( i );
	}
}

void ReadData() {
	bool flag = true;
	int temp_col = 0;
	string line = "", str = "";
	ifstream in( "F://AI/Lab6NN/BPNN_Dataset/train.csv" ); 
	
	while( getline( in, line ) ) {
		temp_col = 0;
		if( flag ) {
			for( int i = 0; i <= line.size(); i ++ ) {
				if( line[ i ] == ',' ) {
					if( temp_col == 2 ) {
						dimension ++;
						attribute.push_back( str );
					}
					else if( temp_col > 3 ) {
						dimension ++;
						attribute.push_back( str );
					}
					str = "";
					temp_col ++;
				}
				else if( i == line.size() )
					str = "";
				else
					str += line[ i ];
		    }
			flag = false;
		}
		else {
			data_num ++;
			for( int i = 0; i <= line.size(); i ++ ) {
				if( line[ i ] == ',' ) {
					if( temp_col == 1 ) 
						dteday.push_back( str );
					else if( temp_col == 2 ) 
						all_data[ attribute[ temp_col - 2 ] ].push_back( convert< double, string >( str ) );
					else if( temp_col > 3 ) 
					    all_data[ attribute[ temp_col - 3 ] ].push_back( convert< double, string >( str ) );
					str = "";
					temp_col ++;
				}
				else if( i == line.size() ) {
					realy_count.push_back( convert< int, string >( str ) );
					str = "";
				}
				else
					str += line[ i ];
			}
		}
	}
	/*for( map< string, vector< double > > :: iterator it = all_data.begin(); it != all_data.end(); it ++ ) 
		if( it -> first != "holiday" && it -> first != "workingday" )
		    it -> second = NormalizeVector( it -> second, it -> first );*/ 
	PartitionDataSet();
	
	in.close();
} 

vector< vector< double > > GetTrainData() {
	vector< double > temp;
	vector< vector< double > > train_data;
	
	train_data.clear();
	for( int i = 0; i < train_id.size(); i ++ ) {
		temp.clear();
		for( int j = 0; j < attribute.size(); j ++ ) 
		    temp.push_back( all_data[ attribute[ j ] ][ train_id[ i ] ] );
		train_data.push_back( temp );
	}
	
	return train_data;
}

vector< vector< double > > GetValidationData() {
	vector< double > temp;
	vector< vector< double > > validation_data;
	
	validation_data.clear();
	for( int i = 0; i < validation_id.size(); i ++ ) {
		temp.clear();
		for( int j = 0; j < attribute.size(); j ++ ) 
		    temp.push_back( all_data[ attribute[ j ] ][ validation_id[ i ] ] );
		validation_data.push_back( temp );
	}
	
	return validation_data;
}

vector< vector< double > > GetAllData() {
	vector< double > temp;
	vector< vector< double > > data;
	
	data.clear();
	for( int i = 0; i < data_num; i ++ ) {
		temp.clear();
		for( int j = 0; j < attribute.size(); j ++ ) 
		    temp.push_back( all_data[ attribute[ j ] ][ i ] );
		data.push_back( temp );
	}
	
	return data;
}

double GetDotProduct( vector< double > vec1, vector< double > vec2, int length ) {
	double result = 0;;
	
	for( int i = 0; i < length; i ++ )
		result += vec1[ i ] * vec2[ i ];
	
	return result; 
}

//结点输入：Input=wx+θ 
double GetInput( vector< double > cur_weight, vector< double > cur_data, double cur_bias, int length ) {
	return GetDotProduct( cur_weight, cur_data, length ) + cur_bias;
}

//激活函数：sigmoid函数：f(x)=1/(1+e^-x) 
double Sigmoid( double x ) {
	return 1.0 / ( 1.0 + exp( -1.0 * x ) );
}

//sigmoid函数的导数：f'(x)=f(x)(1-f(x)) 
double GetDerSigmoid( double x ) {
	return Sigmoid( x ) * ( 1 - Sigmoid( x ) );
}

void Inatial() {  //初始化所有参数 
	vector< double > temp_weight;
	vector< double > temp_step;
	
	n = 0.00001;  //学习率 
	layers = 4;  //层数（4：输入层1+隐藏层2+输出层1） 
	srand( ( unsigned )time( NULL ) );
	for( int i = 0; i < layers; i ++ ) {  //初始化每层结点的数量 
		if( i == 0 )  //输入层：结点数量为特征数量 
		    nodes.push_back( attribute.size() );
		else if( i == layers - 1 )  //输出层：结点数量为1 
		    nodes.push_back( 1 );
	    else  //隐藏层：自行调整 
	        nodes.push_back( 6 );
	}
	for( int i = 0; i < layers - 1; i ++ ) {  
		for( int j = 0; j < nodes[ i + 1 ]; j ++ ) {
			temp_weight.clear();
			temp_step.clear();
			for( int k = 0; k < nodes[ i ]; k ++ ) {
				temp_weight.push_back( ( rand() % 1000 ) * 1.0 / 1001 );
				temp_step.push_back( 0 );
			}
			weights[ i ].push_back( temp_weight );  //权重：随机 
			weight_step[ i ].push_back( temp_step );  //权重步长：0 
			bias[ i ].push_back( ( rand() % 1000 ) * 1.0 / 1001 );  //偏置：随机 
			bias_step[ i ].push_back( 0 );  //偏置步长：0 
		}
	}
}

//结点输出 
vector< double > ForwordPass( vector< double > cur_input, int to_layer ) {
    vector< double > output;
    
    output.clear();
	if( to_layer == layers - 1 )  //输出层：输出=输入 
		output.push_back( cur_input[ 0 ] );
	else  //隐藏层：输出=sigmoid（输入） 
		for( int i = 0; i < nodes[ to_layer ]; i ++ )
			output.push_back( Sigmoid( cur_input[ i ] ) );
	
	return output;
}

//计算梯度误差 
vector< vector< double > > GetErrorGrads( vector< vector< double > > input_data, vector< vector< double > > output_data, int data_id ) {
	double temp_val;
	vector< double > temp_vec;
	vector< vector< double > > error_grads;
	
	error_grads.clear();
	for( int i = layers - 1, cnt = 0; i > 0; i --, cnt ++ ) {
		temp_vec.clear();
		if( i == layers - 1 )  //输出层：err=T-O 
			temp_vec.push_back( realy_count[ data_id ] * 1.0 - output_data[ i ][ 0 ] );
		else {  //隐藏层：err=f'(I)Σerr w 
			for( int j = 0; j < nodes[ i ]; j ++ ) {
				temp_val = 0;
				for( int k = 0; k < nodes[ i + 1 ]; k ++ )
			        temp_val += error_grads[ cnt - 1 ][ k ] * weights[ i ][ k ][ j ];
			    temp_vec.push_back( temp_val * GetDerSigmoid( input_data[ i ][ j ] ) );
			}
		}
		error_grads.push_back( temp_vec );
	}
	
	return error_grads;
}

//更新步长 
void RenewStep( vector< vector< double > > output_data, vector< vector< double > > error_grads ) {
	for( int i = layers - 2, cnt = 0; i >= 0; i --, cnt ++ ) {
		for( int j = 0; j < nodes[ i + 1 ]; j ++ ) {
			for( int k = 0; k < nodes[ i ]; k ++ ) //△w = err O 
				weight_step[ i ][ j ][ k ] = error_grads[ cnt ][ j ] * output_data[ i ][ k ];
			bias_step[ i ][ j ] = error_grads[ cnt ][ j ]; //△θ=err 
		}
	} 
}

//更新权重和偏置 
void RenewWeightAndBias() {
	for( int i = 0; i < layers - 1; i ++ ) {	
		for( int j = 0; j < nodes[ i + 1 ]; j ++ ) {
			for( int k = 0; k < nodes[ i ]; k ++ )  //w = w + η△w 
				weights[ i ][ j ][ k ] += n * weight_step[ i ][ j ][ k ];
			bias[ i ][ j ] += n * bias_step[ i ][ j ];  //θ= θ+ η△θ 
		}
	}
}

//反向传递更新权重 
void BackwardPass( vector< vector< double > > input_data, 
                   vector< vector< double > > output_data, int data_id ) {
	//计算梯度误差，更新步长
	RenewStep( output_data, GetErrorGrads( input_data, output_data, data_id ) );   
	//更新权重
	RenewWeightAndBias();   
}

double GetValidateLoss() {
	double temp_lose = 0;
	vector< vector< double > > validation_data = GetValidationData();
	vector< vector< double > > input_data;
	vector< vector< double > > output_data;
	vector< double > temp_vec;
	
	for( int i = 0; i < validation_id.size(); i ++ ) {
		input_data.clear();
       	output_data.clear();
		input_data.push_back( validation_data[ i ] );
		output_data.push_back( validation_data[ i ] );
		for( int j = 1; j < layers; j ++ ) {
			temp_vec.clear();
			for( int k = 0; k < nodes[ j ]; k ++ )
				temp_vec.push_back( GetInput( weights[ j - 1 ][ k ], output_data[ j - 1 ], bias[ j - 1 ][ k ], nodes[ j - 1 ] ) );
			input_data.push_back( temp_vec );
			output_data.push_back( ForwordPass( input_data[ j ], j ) );
		}
		temp_lose += pow( ( ( int )output_data[ layers - 1 ][ 0 ] - realy_count[ validation_id[ i ] ] ) * 1.0, 2 );
	}
	return temp_lose * 1.0 / validation_id.size();
}

void BPNN() {
	int iterator = 5000;  //迭代次数 
	double temp_lose1 = 0, temp_lose2 = 0;
	vector< vector< double > > train_data = GetTrainData();  //得到训练集数据 
	vector< vector< double > > input_data;  //每层结点的输入
	vector< vector< double > > output_data;  //每层结点的输出 
	vector< double > temp_vec;
	ofstream out1( "train_loss.txt" );
	ofstream out2( "val_loss.txt" );
	
	Inatial();  //初始化各项参数 
	for( int i = 0; i < iterator; i ++ ) {  //迭代到指定次数 
		temp_lose1 = 0;
		temp_lose2 = 0;
		for( int j = 0; j < train_id.size(); j ++ ) {  //每次读取一个训练集样本
			input_data.clear();
        	output_data.clear();
			input_data.push_back( train_data[ j ] );  //输入层：输入为训练样本
			output_data.push_back( train_data[ j ] ); //输出与输入相同 
			for( int k = 1; k < layers; k ++ ) {  //前向传递，分别计算隐藏层和输出层各结点的输入
				temp_vec.clear();
				for( int l = 0; l < nodes[ k ]; l ++ )
				    temp_vec.push_back( GetInput( weights[ k - 1 ][ l ], output_data[ k - 1 ], bias[ k - 1 ][ l ], nodes[ k - 1 ] ) );
				input_data.push_back( temp_vec );  //结点输入 
				output_data.push_back( ForwordPass( input_data[ k ], k ) );  //结点输出 
			}
			temp_lose1 += pow( ( ( int )output_data[ layers - 1 ][ 0 ] - realy_count[ train_id[ j ] ] ) * 1.0, 2 );
			BackwardPass( input_data, output_data, train_id[ j ] );    //反向传递，更新权重 
		}
		temp_lose2 = GetValidateLoss();
		out1 << temp_lose1 * 1.0 / train_id.size() << endl;
		out2 << temp_lose2 << endl;
	}
	out1.close();
	out2.close();
}

void Validate() {
	vector< vector< double > > data = GetAllData();
	vector< vector< double > > input_data;
	vector< vector< double > > output_data;
	vector< double > temp_vec;
	ofstream out( "result.txt" );
	
	for( int i = 0; i < data_num; i ++ ) {
		input_data.clear();
       	output_data.clear();
		input_data.push_back( data[ i ] );
		output_data.push_back( data[ i ] );
		for( int j = 1; j < layers; j ++ ) {
			temp_vec.clear();
			for( int k = 0; k < nodes[ j ]; k ++ )
				temp_vec.push_back( GetInput( weights[ j - 1 ][ k ], output_data[ j - 1 ], bias[ j - 1 ][ k ], nodes[ j - 1 ] ) );
			input_data.push_back( temp_vec );
			output_data.push_back( ForwordPass( input_data[ j ], j ) );
		}
		out << dteday[ i ] << " " 
		    << ( int )output_data[ layers - 1 ][ 0 ] << " " << realy_count[ i ] << endl;
	}
	out.close();
}

void Test() {
	bool flag = true;
	int temp_col = 0;
	int test_num = 0;
	string line = "", str = "";
	vector< double > temp;
	map< string, vector< double > > test_data;
	ifstream in( "F://AI/Lab6NN/BPNN_Dataset/test.csv" ); 
	ofstream out( "15352076_diaohongjin.txt" );
	
	test_data.clear();
	while( getline( in, line ) ) {
		temp_col = 0;
		if( flag ) 
			flag = false;
		else {
			test_num ++;
			for( int i = 0; i <= line.size(); i ++ ) {
				if( line[ i ] == ',' ) {
					if( temp_col == 2 ) 
						test_data[ attribute[ temp_col - 2 ] ].push_back( convert< double, string >( str ) );
					else if( temp_col > 3 ) 
					    test_data[ attribute[ temp_col - 3 ] ].push_back( convert< double, string >( str ) );
					str = "";
					temp_col ++;
				}
				else if( i == line.size() ) 
					str = "";
				else
					str += line[ i ];
			}
		}
	}
	for( map< string, vector< double > > :: iterator it = test_data.begin(); it != test_data.end(); it ++ ) {
		if( it -> first != "holiday" && it -> first != "workingday" ) {
			vector< double > temp2 = it -> second;
			for( int i = 0; i < temp2.size(); i ++ )
			    temp2[ i ] = NormalizeValue( temp2[ i ], it -> first );
			it -> second = temp2;
		}
	}
	vector< double > temp3, temp_vec;
	vector< vector< double > > input_data;
	vector< vector< double > > output_data;
	for( int i = 0; i < test_num; i ++ ) {
		temp3.clear();
		for( int j = 0; j < attribute.size(); j ++ ) 
		    temp3.push_back( test_data[ attribute[ j ] ][ i ] );
		input_data.clear();
       	output_data.clear();
		input_data.push_back( temp3 );
		output_data.push_back( temp3 );
		for( int j = 1; j < layers; j ++ ) {
			temp_vec.clear();
			for( int k = 0; k < nodes[ j ]; k ++ )
				temp_vec.push_back( GetInput( weights[ j - 1 ][ k ], output_data[ j - 1 ], bias[ j - 1 ][ k ], nodes[ j - 1 ] ) );
			input_data.push_back( temp_vec );
			output_data.push_back( ForwordPass( input_data[ j ], j ) );
		}
		out << ( int )output_data[ layers - 1 ][ 0 ] << endl;
	}
	
	in.close();
	out.close();
}

int main() {
	ReadData();
	BPNN();
	Validate();
	//Test();
	
	return 0;
}
