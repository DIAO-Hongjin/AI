#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <sstream> 
#include <cmath>
#include <algorithm>
#include <ctime>
#include <cstdlib>

using namespace std;

template< class out_type, class in_type >  
out_type convert( const in_type & value )   //��������ת�� 
{
    stringstream stream;  
    out_type result;
    
    stream << value;  
    stream >> result; 
    
    return result;
}

struct Tree  //������ 
{
	int root_attr;  //���ڵ�����
	int leaf_label;  //Ҷ�ڵ��ǩ 
	vector< int > attr_value;  //���Ե�ȡֵ 
	vector< struct Tree > children;  //�ӽ�� 
	
	Tree()  //Ĭ�Ϲ��캯�� 
	{
		root_attr = -1;  
		leaf_label = 0;
		attr_value.clear();
		children.clear();
	}
};

struct Cnt  //����ȡֵ������ 
{
	int all_cnt;  //�������Ը�ȡֵ���ֵ��ܴ��� 
	map< int, int > label_cnt;  //�������Ը�ȡֵ��ĳ��ǩ���ֵĴ��� 
	
	void clear()  //���� 
	{
		all_cnt = 0;
		label_cnt.clear();
	}
};

vector< vector< int > > all_data;  //�������ݼ����������� 
vector< int > all_label;  //�������ݼ����������ݶ�Ӧ�ı�ǩ 
set< int > label_type;  //�������ݼ����б�ǩ������
map< int, set< int > > attr_type;  //�������ݵ��������Ե�����ȡֵ 
vector< int > data_id;  //�������ݼ����������ݶ�Ӧ��id 
vector< int > attr_id;  //�������ݼ�����������
vector< int > train_id;  //ѵ�������ݶ�Ӧ��id 
vector< int > validation_id;  //��֤�����ݶ�Ӧ��id 

void ReadALLData()  //��ȡ�������� 
{
	bool flag = true;
	int temp_data_cnt = 0, temp_attr_cnt = 0;
	string line = "", num = "";
	vector< int > temp_vec;
	ifstream in( "F://AI/lab4_Decision_Tree/train.csv" ); 
	
	while( getline( in, line ) ) 
	{
		temp_vec.clear();
		for( int i = 0; i <= line.size(); i ++ )
		{
			if( line[ i ] == ',' )
			{
				temp_vec.push_back( convert< int, string >( num ) );
				num = "";
				if( flag )
					attr_id.push_back( temp_attr_cnt ++ );
			}
			else if( i == line.size() )
			{
				all_label.push_back( convert< int, string >( num ) );
				label_type.insert( convert< int, string >( num ) );
				num = "";
			}
			else
				num += line[ i ];
		}
		all_data.push_back( temp_vec ); 
		data_id.push_back( temp_data_cnt ++ );
		flag = false;
	}
	for( int i = 0; i < attr_id.size(); i ++ )
	    for( int j = 0; j < data_id.size(); j ++ )
	        attr_type[ i ].insert( all_data[ j ][ i ] );
	
	in.close();
}  

void PartitionDataSet()  //����ȡ�����ݼ�����Ϊѵ��������֤�� 
{
	for( int i = 0; i < data_id.size(); i ++ )
	{
		if( i % 10 == 0 )  //��1����������11��������������Ϊ��֤�� 
		    validation_id.push_back( i );  //��֤���洢���������������ݼ��еı�� 
		else  //ʣ����������Ϊѵ���� 
		    train_id.push_back( i );  //ѵ�����洢���������������ݼ��еı�� 
	}
}

//���㵱ǰ���ݼ������Լ��µ�������Ϣ 
//����ֵΪ < ���Ա�ţ�<����ȡֵ��{��ȡֵ���ֵ��ܴ�������ȡֵ��ĳ��ǩ���ֵĴ���} > > 
map< int, map< int, struct Cnt > > GetAttrInfo( vector< int > cur_data_id, vector< int > cur_attr_id )  
{
	int temp_all_cnt = 0;  //ĳһ����ĳһȡֵ���ֵ��ܴ��� 
	struct Cnt temp_cnt;  //�洢����ĳ��ȡֵ���ֵ��ܴ����Լ���ĳ����ǩ���ֵĴ�������ʱ���� 
	map< int, int > temp_label_cnt;  //ĳһ����ĳһȡֵ��ĳ����ǩ���ֵĴ��� 
	map< int, struct Cnt > temp_attr;  //��ǰ���ݼ���ĳһ���Ե���Ϣ 
	map< int, map< int, struct Cnt > > attr_info;  //��ǰ���ݼ����������Ե���Ϣ 
	
	attr_info.clear();
	for( int i = 0; i < cur_attr_id.size(); i ++ )  //������ǰ���ݼ���ÿһ������ 
	{
		temp_attr.clear();
		//���������Ե�ÿ��ȡֵ
		for( set< int > :: iterator it = attr_type[ cur_attr_id[ i ] ].begin(); it != attr_type[ cur_attr_id[ i ] ].end(); it ++ )   
		{
			temp_all_cnt = 0;
			temp_label_cnt.clear();
			temp_cnt.clear();
			//������ǰ���ݼ���ͳ�ƴ���
			for( int j = 0; j < cur_data_id.size(); j ++ )   
			{
				//�����Եĸ�ȡֵ���ִ�������  
				if( all_data[ cur_data_id[ j ] ][ cur_attr_id[ i ] ] == *it )  
			    {
			    	temp_all_cnt ++;  
			    	//�������б�ǩ
			    	for( set< int > :: iterator temp_it = label_type.begin(); temp_it != label_type.end(); temp_it ++ ) 
						//����ȡֵ�ڸñ�ǩ�ĳ��ִ�������  
			    	    if( ( *temp_it ) == all_label[ cur_data_id[ j ] ] )   
			    	        temp_label_cnt[ *temp_it ] ++;   
			    }
			}
			temp_cnt.all_cnt = temp_all_cnt; 
			temp_cnt.label_cnt = temp_label_cnt;
			temp_attr[ *it ] = temp_cnt;  //����¼��ֵ�������ݽṹ�� 
		}
		attr_info[ cur_attr_id[ i ] ] = temp_attr;  //��¼�����Ե�������Ϣ 
	}
	
	return attr_info;  //�����������Ե���Ϣ 
}

double GetHD( vector< int > cur_data_id )  //���㵱ǰ���ݼ��ľ����� 
{
	double hd = 0;
	map< int, int > label_sum;
	
	label_sum.clear();
	//���㵱ǰ���ݼ��и���ǩ�ĳ��ִ��� 
	for( set< int > :: iterator it = label_type.begin(); it != label_type.end(); it ++ )
	{
		for( int i = 0; i < cur_data_id.size(); i ++ )
	    {
	    	if( all_label[ cur_data_id[ i ] ] == ( *it ) )
	            label_sum[ *it ] ++;
	        else
	            label_sum[ *it ] += 0;
	    }
	}
	//���ݹ�ʽ����H(D) 
	for( map< int, int > :: iterator it = label_sum.begin(); it != label_sum.end(); it ++ )
	    hd -= ( ( it -> second ) * 1.0 / cur_data_id.size() ) * ( log( ( ( it -> second ) * 1.0 / cur_data_id.size() ) ) / log( 2 ) );
	
	return hd;  //�������ݼ��ľ����� 
}

//���㵱ǰ���ݼ��ڵ�ǰ���������µ������� 
double GetHDA( vector< int > cur_data_id, map< int, map< int, struct Cnt > > attr_info, int current_attr )
{
	double hda = 0, temp = 0;
	
	//���ݹ�ʽ���㵱ǰ���Ե������� H(D|A)
	for( map< int, struct Cnt > :: iterator it1 = attr_info[ current_attr ].begin(); it1 != attr_info[ current_attr ].end(); it1 ++ )
	{
		temp = 0;
		for( map< int, int > :: iterator it2 = ( it1 -> second ).label_cnt.begin(); it2 != ( it1 -> second ).label_cnt.end(); it2 ++ )
			temp -= ( it2 -> second ) * 1.0 / ( ( it1 -> second ).all_cnt ) * ( log( ( it2 -> second ) * 1.0 / ( ( it1 -> second ).all_cnt ) ) / log( 2 ) );
		temp *= ( ( it1 -> second ).all_cnt ) * 1.0 / cur_data_id.size();
		hda += temp;
	}
	
	return hda;  //�������ݼ��ڵ�ǰ���������µ������� 
}

//���㵱ǰ���ݼ��ڵ�ǰ���������µ���Ϣ���� 
double GetGDA( vector< int > cur_data_id, map< int, map< int, struct Cnt > > attr_info, int current_attr )
{
	//g(D,A) = H(D)-H(D|A) 
	double gda = GetHD( cur_data_id ) - GetHDA( cur_data_id, attr_info, current_attr );
	
	return gda;  //�������ݼ��ڵ�ǰ���������µ���Ϣ���� 
}


//���㵱ǰ���ݼ���ǰ���Ե��� 
double GetSplitInfoDA( vector< int > cur_data_id, map< int, map< int, struct Cnt > > attr_info, int current_attr )
{
	double sida = 0;
	
	for( map< int, struct Cnt > :: iterator it = attr_info[ current_attr ].begin(); it != attr_info[ current_attr ].end(); it ++ )
	    //�����Եĸ�ȡֵ�ڵ�ǰ���ݼ��г��ִ���Ϊ0������Ϊ0��������ֶ����������������� 
	    if( ( it -> second ).all_cnt == 0 )
	        sida -= 0;
	    else  //���ݹ�ʽ����SplitInfo(D,A) 
		    sida -= ( ( ( it -> second ).all_cnt ) * 1.0 / cur_data_id.size() ) * ( log( ( ( it -> second ).all_cnt ) * 1.0 / cur_data_id.size() ) / log( 2 ) );
	
	return sida;  //���ص�ǰ���ݼ������Ե��� 
}

//������Ϣ������ 
double GetGRatioA( vector< int > cur_data_id, vector< int > cur_attr_id, map< int, map< int, struct Cnt > > attr_info, int current_attr )  
{
	double gra = 0, avg_sida = 0;
	
	//�����������Ե��ص�ƽ��ֵ 
	for( int i = 0; i < cur_attr_id.size(); i ++ )
	    avg_sida += GetSplitInfoDA( cur_data_id, attr_info, cur_attr_id[ i ] );
	avg_sida /= 1.0 * cur_attr_id.size();
	//������Ϣ�����ʣ���ĸ�����������Ե��ص�ƽ��ֵ����ƽ����������ַ�ĸΪ0����� 
	gra = GetGDA( cur_data_id, attr_info, current_attr ) / ( avg_sida + GetSplitInfoDA( cur_data_id, attr_info, current_attr ) );
	
	return gra;  //�������Ե���Ϣ������ 
}

//�������ָ�� 
double GetGiniDA( vector< int > cur_data_id, map< int, map< int, struct Cnt > > attr_info, int current_attr )   
{
	double ginida = 0, temp = 0;
	
	//���ݹ�ʽ�������ָ�� 
	for( map< int, struct Cnt > :: iterator it1 = attr_info[ current_attr ].begin(); it1 != attr_info[ current_attr ].end(); it1 ++ )
	{
		temp = 1;
		for( map< int, int > :: iterator it2 = ( it1 -> second ).label_cnt.begin(); it2 != ( it1 -> second ).label_cnt.end(); it2 ++ )
			temp -= pow( ( it2 -> second ) * 1.0 / ( ( it1 -> second ).all_cnt ), 2 );
		temp *= ( ( it1 -> second ).all_cnt ) * 1.0 / cur_data_id.size();
		ginida += temp;
	}
	
	return ginida;  //���ػ���ָ�� 
}

set< int > FindLabelType( vector< int > cur_data_id )
{
	set< int > result_set;
	
	result_set.clear();
	for( int i = 0; i < cur_data_id.size(); i ++ )
		result_set.insert( all_label[ cur_data_id[ i ] ] );
	
	return result_set;
}

//ѡȡ���ֵ��ý��������г��ִ������ı�ǩ��ΪҶ�ӽ��ı�ǩ 
int ChooseLabel( vector< int > cur_data_id )
{
	int max = 0, result = 0;
	map< int, int > cur_label_cnt;
	set< int > cur_label_set = FindLabelType( cur_data_id );
	
	cur_label_cnt.clear();
	//ͳ��ÿ����ǩ�ĳ��ִ��� 
	for( set< int > :: iterator it = cur_label_set.begin(); it != cur_label_set.end(); it ++ )
		cur_label_cnt[ *it ] ++;
	//�ҵ����ִ������ı�ǩ 
	for( map< int, int > :: iterator it = cur_label_cnt.begin(); it != cur_label_cnt.end(); it ++ )
	{
		if( max < ( it -> second ) )
		{
			max = it -> second;
			result = it -> first;
		}
	}
	
	return result;  //����ѡ��ı�ǩ 
}

//ѡ����ߵ������ 
int ChooseRootAttr( string way, vector< int > cur_data_id, vector< int > cur_attr_id )
{
	int root_attr = 0;
	double max = -10000, min = 10000, temp = 0;
	map< int, map< int, struct Cnt > > attr_info = GetAttrInfo( cur_data_id, cur_attr_id );
	
	if( way == "ID3" )  //ID3�㷨��ѡ����Ϣ������������ 
	{
		for( int i = 0; i < cur_attr_id.size(); i ++ )
		{
			temp = GetGDA( cur_data_id, attr_info, cur_attr_id[ i ] );
			if( max < temp )
		    {
	    		max = temp;
	    		root_attr = cur_attr_id[ i ];
		    }
		}
	}
	else if( way == "C4.5" )  //C4.5�㷨��ѡ����Ϣ�������������� 
	{
		for( int i = 0; i < cur_attr_id.size(); i ++ )
		{
			temp = GetGRatioA( cur_data_id, cur_attr_id, attr_info, cur_attr_id[ i ] );
			if( max < temp )
		    {
	    		max = temp;
	    		root_attr = cur_attr_id[ i ];
		    }
		}
	}
	else if( way == "CART" )  //CART�㷨��ѡ�����ϵ����С������ 
	{
		for( int i = 0; i < cur_attr_id.size(); i ++ )
		{
			temp = GetGiniDA( cur_data_id, attr_info, cur_attr_id[ i ] );
			if( min > temp )
		    {
	    		min = temp;
	    		root_attr = cur_attr_id[ i ];
		    }
		}
	}
	
	return root_attr;  //����ѡ������� 
}

void CreateTree( string way, vector< int > far_data_id, vector< int > cur_data_id, vector< int > cur_attr_id, struct Tree &tree )
{
	vector< int > new_data_id, new_attr_id;
	set< int > res_label_type = FindLabelType( cur_data_id );
	
	if( res_label_type.size() == 1 )
	{
		for( set< int > :: iterator it = res_label_type.begin(); it != res_label_type.end(); it ++ )
		    tree.leaf_label = *it;
		return ;
	}
	if( cur_data_id.size() == 0 )
	{
		tree.leaf_label = ChooseLabel( far_data_id );
		return ;
	}
	if( cur_attr_id.size() == 0 )
	{
		tree.leaf_label = ChooseLabel( cur_data_id );
		return ;
	}
	tree.root_attr = ChooseRootAttr( way, cur_data_id, cur_attr_id );
	for( set< int > :: iterator it = attr_type[ tree.root_attr ].begin(); it != attr_type[ tree.root_attr ].end(); it ++ )
	{
		new_data_id.clear();
		new_attr_id.clear();
		for( int j = 0; j < cur_attr_id.size(); j ++ )
		    if( cur_attr_id[ j ] != tree.root_attr )
		        new_attr_id.push_back( cur_attr_id[ j ] );
		for( int j = 0; j < cur_data_id.size(); j ++ )
		    if( all_data[ cur_data_id[ j ] ][ tree.root_attr ] == ( *it ) )
		        new_data_id.push_back( cur_data_id[ j ] ); 
		struct Tree new_tree;
		CreateTree( way, cur_data_id, new_data_id, new_attr_id, new_tree );
		tree.attr_value.push_back( *it );
		tree.children.push_back( new_tree );
	} 
}

int PredictLabel( vector< int > data, struct Tree tree )
{
	if( tree.root_attr == -1 )
	    return tree.leaf_label;
	
	for( int i = 0; i < tree.attr_value.size(); i ++ )
	    if( data[ tree.root_attr ] == tree.attr_value[ i ] )
	        return PredictLabel( data, tree.children[ i ] );
	
	return *( label_type.begin() );//������˵���������ݵ�ĳ�����Գ�����ѵ�����в����ڵ�ȡֵ������ָ����ǩ
}

vector< struct Tree > CreateForest( string way, int size )  //�������ɭ�� 
{
	vector< struct Tree > forest;
	vector< int > radom_data_id;
	
	forest.clear();
	for( int i = 0; i < size; i ++ )  //����size�þ����� 
	{
		struct Tree tree;
		srand( ( unsigned )time( NULL ) );
		radom_data_id.clear();
		for( int j = 0; j < train_id.size(); j ++ )  //�зŻ���������������ѵ���� 
		    radom_data_id.push_back( train_id[ rand() % train_id.size() ] );
		CreateTree( way, radom_data_id, radom_data_id, attr_id, tree );  //���������� 
		forest.push_back( tree );  //������������ɭ���� 
	}
	
	return forest;  //���ش����õ����ɭ�� 
}

//�õ�����Ԥ���ǩ
int PredictFinalLabel( vector< int > data, vector< struct Tree > forest )   
{
	int max = 0, label = 0;
	map< int, int > label_cnt;
	
	label_cnt.clear();
	for( int i = 0; i < forest.size(); i ++ )  //����ÿһ����Ԥ���ǩ��ͳ��ÿ����ǩ������ 
	    label_cnt[ PredictLabel( data, forest[ i ] ) ] ++;
	//�ҵ����ִ������ı�ǩ 
	for( map< int, int > :: iterator it = label_cnt.begin(); it != label_cnt.end(); it ++ )
	{
		if( max < ( it -> second ) )
		{
			max = it -> second;
			label = it -> first;
		}
	}
	
	return label;  //���س������ı�ǩ��Ϊ����Ԥ��ı�ǩ 
}

void Validation( string way, vector< int > validation_id, vector< struct Tree > forest )
{
	int cnt = 0;
	
	for( int i = 0; i < validation_id.size(); i ++ )
	{
		if( all_label[ validation_id[ i ] ] == PredictFinalLabel( all_data[ validation_id[ i ] ], forest ) )
			cnt ++;
	}
	
	cout << way << ": ��ȷ��Ϊ" << cnt * 1.0 / validation_id.size() << endl;
}

int main()
{
	ReadALLData();
	PartitionDataSet();
	
	for( int i = 0; i < 50; i ++ )
	{
		vector< struct Tree > ID3_forest = CreateForest( "ID3", 25 );
		vector< struct Tree > C45_forest = CreateForest( "C4.5", 25 );
		vector< struct Tree > CART_forest = CreateForest( "CART", 25 );
	
		Validation( "ID3", validation_id, ID3_forest );
		Validation( "C4.5", validation_id, C45_forest );
		Validation( "CART", validation_id, CART_forest );
		
		cout << endl;
	} 
	
	return 0;
}


