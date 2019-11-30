#include <iostream>
#include <algorithm>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <map> 
#include <set>

using namespace std;

int train_text_num, train_word_num;  //ѵ�����ݵ��ı������͵�������
map< string, double > train_label_prob;
map< string, int > train_label_sum;
map< string, set< string > > train_label_word;
map< pair< string, string >, int > train_word_frq;
map< int, string > train_label;  //ѵ�������ı���Ӧ�ı�ǩ
map< string, int > train_vocabulary;  //��¼ѵ�����������е��ʳ��ֵ��Ⱥ�˳�� 
vector< vector< double > > train_tf; 
vector< vector< string > > train_data;  //��¼ѵ��������ÿƪ���³�����ʲô����
 
void SplitString()  
{
	int end_pos = 0;
	string line = "", word = "", label = "";
	vector< string > temp_vec;  
	map< string, int > train_label_num;
	ifstream in( "F://AI/lab2(KNN+NB)/DATA/classification_dataset/train_set.csv" ); 
	
	train_label_num.clear();
	getline( in, line );
	while( getline( in, line ) )  //���ж�ȡ�ļ� 
	{
		end_pos = 0;
		word = "";
		label = "";
		temp_vec.clear();
		for( int i = 0; i < line.size(); i ++ ) 
			if( line[ i ] == ',' )  
		    	end_pos = i;
		for( int i = end_pos + 1; i < line.size(); i ++ )
            label += line[ i ];
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
        		    temp_vec.push_back( word );
        		    train_label_word[ label ].insert( word );
        		}
        		word = "";
        	}
        }
		train_data.push_back( temp_vec );  
		train_label[ train_text_num ++ ] = label;
		train_label_num[ label ] ++;
		train_label_sum[ label ] += temp_vec.size();
	}
	for( map< string, int > :: iterator it = train_label_num.begin(); it != train_label_num.end(); it ++ )
	    train_label_prob[ it -> first ] = ( it -> second ) * 1.0 / train_text_num;
	  
	in.close();
}

void GetTF()  
{
	for( int i = 0; i < train_text_num; i ++ )  //����ѵ�����ı� 
		for( int j = 0; j < train_data[ i ].size(); j ++ )  //����ѵ�����ı��е�ÿһ������  
			train_word_frq[ pair< string, string >( train_data[ i ][ j ], train_label[ i ] ) ] ++;  //��¼��������������ǩ�³��ֵĴ���
}

string NBModel( vector< string > test_data )
{
	double a = 0.55;
	double max = -10000000;
	string result = "";
	map< string, double > test_prob;
	
	test_prob.clear();
	for( int i = 0; i < test_data.size(); i ++ )
	{
	    for( map< string, int > :: iterator it = train_label_sum.begin(); it != train_label_sum.end(); it ++ )
	    {
	    	if( train_vocabulary[ test_data[ i ] ] > 0 )
	    		test_prob[ it -> first ] += log( ( train_word_frq[ pair< string, string >( test_data[ i ], it -> first ) ] + a ) * 1.0 / ( it -> second + a * train_word_num ) );
	    	else
	    	    test_prob[ it -> first ] += 0;
	    }
	}
	for( map< string, double > :: iterator it = test_prob.begin(); it != test_prob.end(); it ++ )
	    it -> second += log( train_label_prob[ it -> first ] );
	for( map< string, double > :: iterator it = test_prob.begin(); it != test_prob.end(); it ++ )
	{
		if( max < ( it -> second ) )
	    {
	    	max = ( it -> second );
	        result = it -> first;
	    }
	}
	
	return result;
}

void OptimizeModel()
{
	int end_pos = 0, cnt = 0, sum = 0;
	string line = "", word = "", label = "";
	vector< string > temp;  
	ifstream in( "F://AI/lab2(KNN+NB)/DATA/classification_dataset/validation_set.csv" ); 

	getline( in, line );
	while( getline( in, line ) )  //���ж�ȡ�ļ� 
	{
		sum ++;
		end_pos = 0;
		word = "";
		label = "";
		temp.clear();
		for( int i = 0; i < line.size(); i ++ ) 
			if( line[ i ] == ',' )  
		    	end_pos = i;
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
        for( int i = end_pos + 1; i < line.size(); i ++ )
            label += line[ i ];
        if( NBModel( temp ) == label )
            cnt ++;
	}
	cout << "��֤���ı�������" << sum << " Ԥ����ȷ�ı�ǩ������" << cnt << endl;
	cout << "׼ȷ�ʣ� " << 1.0 * cnt / sum << endl;
	  
	in.close();
}

void PredictLabel() 
{
	bool flag = false;
	int start_pos = 0, end_pos = 0, cnt = 0, textid = 0;
	string line = "", word = "";
	vector< string > temp;  
	ifstream in( "F://AI/lab2(KNN+NB)/DATA/classification_dataset/test_set.csv" ); 
	ofstream out( "15352076_diaohongjin_NB_classification.csv" );

	getline( in, line );
	out << "textid,label" << endl;
	while( getline( in, line ) )  //���ж�ȡ�ļ� 
	{
		textid ++;
		flag = false;
		start_pos = 0;
		end_pos = 0;
		word = "";
		temp.clear();
		for( int i = 0; i < line.size(); i ++ ) 
		{
			if( line[ i ] == ',' && ! flag ) 
			{
				start_pos = i + 1;
				flag = true;
			} 
			else if( line[ i ] == ',' && flag )
		    	end_pos = i;
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
        out << textid << "," << NBModel( temp ) << endl;
	}
	  
	in.close();
	out.close();
}

int main()
{
	int choice;
	 
	SplitString();
	GetTF();
	
	cout << "��ȡ��֤��������1����ȡ���Լ��밴2" << endl << "����ѡ���ǣ�";
	cin >> choice;	
	if( choice == 1 )
	    OptimizeModel();
	if( choice == 2 )
		PredictLabel();

	return 0;
} 
