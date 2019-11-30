#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <map> 

using namespace std;

int train_text_num, train_word_num;  //ѵ�����ݵ��ı������͵������� 
map< string, int > train_vocabulary;  //��¼ѵ�����������е��ʳ��ֵ��Ⱥ�˳�� 
vector< vector< double > > train_prob;  //��¼ѵ���������ı��������ĸ��� 
vector< vector< double > > train_tf;  //ѵ�������ı���tf����
vector< vector< string > > train_data;  //��¼ѵ��������ÿƪ���³�����ʲô����
vector< vector< double > > predict_prob;  //��¼ѵ���������ı��������ĸ��� 

template< class out_type, class in_type >  
out_type convert( const in_type & value )  //�������������໥ת�� 
{
    stringstream stream;  //����һ���� 
    
    out_type result;  //�洢ת�����
    stream << value;  //�����д�ֵ
    stream >> result;  //��result��д��ֵ
    
    return result;  //����ת�����
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
	while( getline( in, line ) )  //���ж�ȡ�ļ� 
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

void GetTF()  
{
	int cnt = 0;
	vector< double > temp; 
    
	for( int i = 0; i < train_text_num; i ++ ) 
	{
		temp.clear();
		for( int j = 0; j < train_word_num; j ++ )  
	    {
	    	cnt = 0;
	    	for( int k = 0; k < train_data[ i ].size(); k ++ )  
	        	if( train_vocabulary[ train_data[ i ][ k ] ] - 1 == j )
	        		cnt ++;
	        temp.push_back( 1.0 * cnt );   
	    }
	    train_tf.push_back( temp );
	}
}

void NBModel( vector< string > test_data )
{
	double a = 0.035;
	double temp_prob = 0, sum_prob = 0;
	vector< double > label_prob;
	
	for( int i = 0; i < 6; i ++ )
	    label_prob.push_back( 0 );
	for( int i = 0; i < train_text_num; i ++ ) 
	{
		temp_prob = 1;
		for( int j = 0; j < test_data.size(); j ++ )
		{
			if( train_vocabulary[ test_data[ j ] ] > 0 )
				temp_prob *= ( train_tf[ i ][ train_vocabulary[ test_data[ j ] ] - 1 ] + a ) * 1.0 / ( train_data[ i ].size() + a * train_word_num );
			else
				temp_prob *= 1;
		}
		for( int j = 0; j < 6; j ++ )
		    label_prob[ j ] += temp_prob * train_prob[ i ][ j ];
	}
	for( int i = 0; i < 6; i ++ )
	    sum_prob += label_prob[ i ];
	for( int i = 0; i < 6; i ++ )
	    label_prob[ i ] /= sum_prob;
	predict_prob.push_back( label_prob );
} 

void OptimizeModel()
{
	bool flag = false;
	int end_pos = 0, cnt = 0, sum = 0;
	string line = "", word = "";
	vector< string > temp;  
	ifstream in( "F://AI/lab2(KNN+NB)/DATA/regression_dataset/validation_set.csv" ); 
	
	getline( in, line );
	while( getline( in, line ) )  //���ж�ȡ�ļ� 
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
        NBModel( temp );
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
	while( getline( in, line ) )  //���ж�ȡ�ļ� 
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
        NBModel( temp );
	}
	  
	in.close();
}

int main()
{
	int choice;
	
	SplitString();
	GetTF();
	
	cout << "��ȡ��֤��������1����ȡ���Լ��밴2" << endl << "����ѡ���ǣ�";
	cin >> choice;
	if( choice == 1 )
	{
		OptimizeModel();
		ofstream out( "��֤��NB�ع���.txt" );
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
		ofstream out( "15352076_diaohongjin_NB_regression.csv" );
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
