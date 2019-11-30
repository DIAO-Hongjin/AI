#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <map> 

using namespace std;

int train_text_num, train_word_num;  //训练数据的文本数量和单词总数 
map< string, int > train_vocabulary;  //记录训练数据中所有单词出现的先后顺序 
vector< vector< double > > train_prob;  //记录训练数据中文本各情绪的概率 
vector< double > idf;  //idf
vector< vector< double > > train_tf_idf;  //训练数据文本的tf-idf矩阵
vector< vector< string > > train_data;  //记录训练数据中每篇文章出现了什么单词
vector< vector< double > > predict_prob;  //记录训练数据中文本各情绪的概率 

template< class out_type, class in_type >  
out_type convert( const in_type & value )  //任意两种类型相互转换 
{
    stringstream stream;  //创建一个流 
    
    out_type result;  //存储转换结果
    stream << value;  //向流中传值
    stream >> result;  //向result中写入值
    
    return result;  //返回转换结果
}

void SplitString()  
{
	int end_pos = 0;
	double prob = 0;
	bool flag = false;
	string line = "", word = "", num = "";
	vector< string > temp_word;
	vector< double > temp_prob;  
	ifstream in( "F://AI/lab2(KNN+NB)/DATA/regression_dataset/train_set.csv" ); 
	
	getline( in, line );
	while( getline( in, line ) )  //按行读取文件 
	{
		end_pos = 0;
		prob = 0;
		word = "";
		num = "";
		flag = false;
		temp_word.clear();
		temp_prob.clear();
		for( int i = 0; i < line.size(); i ++ ) 
		{
			if( line[ i ] == ',' && ! flag )  
		    {
		    	end_pos = i;
		    	flag = true;
		    }
		}
        for( int i = 0; i <= end_pos; i ++ ) 
        {
        	if( line[ i ] != ' ' && i != end_pos ) 
        	    word += line[ i ];
        	else
        	{
        		if( word != "" )
        		{
        			if( train_vocabulary[ word ] == 0 )   
        			    train_vocabulary[ word ] = ++ train_word_num;  
        		    temp_word.push_back( word );
        		}
        		word = "";
        	}
        }
        for( int i = end_pos + 1; i <= line.size(); i ++ )
        {
        	if( line[ i ] != ',' && i != line.size() ) 
        	    num += line[ i ];
        	else
        	{
        		if( num != "" )
        			temp_prob.push_back( convert< double, string >( num ) );
        		num = "";
        	}
        }
		train_data.push_back( temp_word ); 
		train_prob.push_back( temp_prob ); 
		train_text_num ++;
	}
	  
	in.close();
}

void GetTFIDFMatrix()  
{
	int cnt = 0;
	vector< double > temp; 
    
    for( int i = 0; i < train_word_num; i ++ )  
    {
    	cnt = 0;
    	for( int j = 0; j < train_text_num; j ++ ) 
        {
        	for( int k = 0; k < train_data[ j ].size(); k ++ )   
	        {
	        	if( train_vocabulary[ train_data[ j ][ k ] ] - 1 == i )
	        	{
	        		cnt ++;   
	        		break;
	        	}
	        }
        }
	    idf.push_back( log( 1.0 * train_text_num / ( 1 + cnt ) ) / log( 2 ) );   
    }
	for( int i = 0; i < train_text_num; i ++ ) 
	{
		temp.clear();
		for( int j = 0; j < train_word_num; j ++ )  
	    {
	    	cnt = 0;
	    	for( int k = 0; k < train_data[ i ].size(); k ++ )  
	        	if( train_vocabulary[ train_data[ i ][ k ] ] - 1 == j )
	        		cnt ++;
	        temp.push_back( 1.0 * cnt * idf[ j ] / train_data[ i ].size() );   
	    }
	    train_tf_idf.push_back( temp );
	}
}

double GetDistance( vector< double > train_data_vec, vector< double > test_data_vec )
{
	double cos_distance = 0, train_length = 0, test_length = 0, temp = 0;
	
	if( train_data_vec.size() < test_data_vec.size() )
		for( int i = train_data_vec.size(); i < test_data_vec.size(); i ++ )
		    train_data_vec.push_back( 0 );
	
	for( int i = 0; i < test_data_vec.size(); i ++ )
	{
		train_length += pow( 1.0 * train_data_vec[ i ], 2 );
		test_length += pow( 1.0 * test_data_vec[ i ], 2 );
		temp += 1.0 * train_data_vec[ i ] * test_data_vec[ i ];
	}
	return 1.001 - temp / ( sqrt( train_length ) * sqrt( test_length ) );
}

vector< double > NormailizeVector( multimap< double, int > dis, int k )
{
	int cnt = 0;
	vector< double > result;
	double max = -1000000, min = 1000000;
	
	result.clear();
	for( multimap< double, int > :: iterator it = dis.begin(); cnt < k; cnt ++, it ++ )
	{ 
		if( min >= 1 / ( * it ).first )
		    min = 1 / ( * it ).first;
		if( max <= 1 / ( * it ).first )
		    max = 1 / ( * it ).first;
	}
	cnt = 0;
	if( max - min == 0 )
	    for( multimap< double, int > :: iterator it = dis.begin(); cnt < k; cnt ++, it ++ )
		    result.push_back( 1 / ( * it ).first );
	else
	    for( multimap< double, int > :: iterator it = dis.begin(); cnt < k; cnt ++, it ++ )
		    result.push_back( ( 1 / ( * it ).first - min ) / ( max - min ) );
	    
	return result;
}

void KNNModel( vector< string > test_data )
{
	int k = 17, cnt = 0, max = 0;
	int temp_word_num = train_word_num;
	bool flag = false;
	double sum = 0;
	map< string, int > temp_voc;
	multimap< double, int > vec_distance;
	vector< double > temp_prob;
	vector< double > test_vec;
	vector< double > temp_idf;
	vector< double > dis;
	
	temp_voc = train_vocabulary;
	temp_idf = idf;
	vec_distance.clear();
	test_vec.clear();
	dis.clear();
	for( int i = 0; i < 6; i ++ )
	    temp_prob.push_back( 0 );
	for( int i = 0; i < test_data.size(); i ++ )
	{
		if( temp_voc[ test_data[ i ] ] == 0 )   
        {
        	temp_voc[ test_data[ i ] ] = ++ temp_word_num; 
            temp_idf.push_back( log( 1.0 * ( train_text_num + 1 ) / 2 ) / log( 2 ) ); 
        }
	}
	for( int i = 0; i < temp_word_num; i ++ )  
	{
	    for( int j = 0; j <  test_data.size(); j ++ ) 
	        if( temp_voc[ test_data[ j ] ] - 1 == i )
	        	cnt ++;
	    test_vec.push_back( 1.0 * cnt * temp_idf[ i ] / test_data.size() );
	    cnt = 0;
	}
	for( int i = 0; i < train_text_num; i ++ ) 
		vec_distance.insert( pair< double, int >( GetDistance( train_tf_idf[ i ], test_vec ), i ) );
	dis = NormailizeVector( vec_distance, k );  
	for( multimap< double, int > :: iterator it = vec_distance.begin(); cnt < k; cnt ++, it ++ )
	{
		temp_prob[ 0 ] += train_prob[ ( * it ).second ][ 0 ] * dis[ cnt ];
		temp_prob[ 1 ] += train_prob[ ( * it ).second ][ 1 ] * dis[ cnt ];
		temp_prob[ 2 ] += train_prob[ ( * it ).second ][ 2 ] * dis[ cnt ];
		temp_prob[ 3 ] += train_prob[ ( * it ).second ][ 3 ] * dis[ cnt ];
		temp_prob[ 4 ] += train_prob[ ( * it ).second ][ 4 ] * dis[ cnt ];
		temp_prob[ 5 ] += train_prob[ ( * it ).second ][ 5 ] * dis[ cnt ];
	}
	sum = temp_prob[ 0 ] + temp_prob[ 1 ] + temp_prob[ 2 ] + temp_prob[ 3 ] + temp_prob[ 4 ] + temp_prob[ 5 ];
	temp_prob[ 0 ] /= sum;
	temp_prob[ 1 ] /= sum;
	temp_prob[ 2 ] /= sum;
	temp_prob[ 3 ] /= sum;
	temp_prob[ 4 ] /= sum;
	temp_prob[ 5 ] /= sum;
	predict_prob.push_back( temp_prob );
}

void OptimizeModel()
{
	bool flag = false;
	int end_pos = 0, cnt = 0, sum = 0;
	string line = "", word = "";
	vector< string > temp;  
	ifstream in( "F://AI/lab2(KNN+NB)/DATA/regression_dataset/validation_set.csv" ); 
	
	getline( in, line );
	while( getline( in, line ) )  //按行读取文件 
	{
		end_pos = 0;
		word = "";
		flag = false;
		temp.clear();
		for( int i = 0; i < line.size(); i ++ ) 
		{
			if( line[ i ] == ',' && ! flag )  
		    {
		    	end_pos = i;
		    	flag = true;
		    }
		}
        for( int i = 0; i <= end_pos; i ++ ) 
        {
        	if( line[ i ] != ' ' && i != end_pos ) 
        	    word += line[ i ];
        	else
        	{
        		if( word != "" ) 
        		    temp.push_back( word );
        		word = "";
        	}
        }
        KNNModel( temp );
	}
	
	in.close();
}

void PredictLabel() 
{
	int start_pos = 0, end_pos = 0, cnt = 0, textid = 0;
	string line = "", word = "";
	vector< string > temp;  
	ifstream in( "F://AI/lab2(KNN+NB)/DATA/regression_dataset/test_set.csv" ); 
	
	getline( in, line );
	while( getline( in, line ) )  //按行读取文件 
	{
		textid ++;
		cnt = 0;
		start_pos = 0;
		end_pos = 0;
		word = "";
		temp.clear();
		for( int i = 0; i < line.size(); i ++ ) 
		{
			if( line[ i ] == ',' && cnt == 0 ) 
			{
				start_pos = i + 1;
				cnt ++;
			} 
			else if( line[ i ] == ',' && cnt == 1 )
		    {
		    	end_pos = i;
		    	cnt ++;
		    }
		}
        for( int i = start_pos; i <= end_pos; i ++ ) 
        {
        	if( line[ i ] != ' ' && i != end_pos ) 
        	    word += line[ i ];
        	else
        	{
        		if( word != "" )
        		    temp.push_back( word );
        		word = "";
        	}
        }
        KNNModel( temp );
	}
	  
	in.close();
}

int main()
{
	int choice;
	 
	SplitString();
	GetTFIDFMatrix();
	
	cout << "读取验证集请输入1，读取测试集请按2" << endl << "您的选择是：";
	cin >> choice;
	if( choice == 1 )
	{
		OptimizeModel();
		ofstream out( "验证集KNN回归结果.txt" );
    	for( int i = 0; i < 311; i ++ )
	    {
		    for( int j = 0; j < 6; j ++ )
			    out << predict_prob[ i ][ j ] << "\t";
    	    out << endl;
	    }
    	out.close();
	}
	if( choice == 2 )
	{
		PredictLabel();
		ofstream out( "15352076_diaohongjin_KNN_regression.csv" );
    	out << "textid,anger,disgust,fear,joy,sad,surprise" << endl;
	    for( int i = 0; i < 312; i ++ )
    	{
	    	out << i + 1; 
		    for( int j = 0; j < 6; j ++ )
			    out << "," << predict_prob[ i ][ j ];
    	    out << endl;
	    }
    	out.close();
	} 
	
	return 0;
}
