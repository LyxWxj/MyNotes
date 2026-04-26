## 数据格式读取：GDAL/OGR
### 依赖
![[Pasted image 20250115135229.png]]

### OGR矢量数据支持（已去除没有地理参考的格式）
#### 常用矢量格式

| 格式名            | 扩展名       | 使用市场占比 | 主要用途                          |
| -------------- | --------- | ------ | ----------------------------- |
| ESRI Shapefile | .shp      | 35%    | 地理空间数据存储与交换，广泛应用于GIS软件        |
| GeoJSON        | .geojson  | 20%    | Web地图开发、地理数据分析、位置服务           |
| KML/KMZ        | .kml/.kmz | 15%    | 地理信息展示，如Google Earth          |
| CAD (DWG/DXF)  | .dwg/.dxf | 10%    | 测绘数据采集与交换，广泛用于CAD软件           |
| GeoPackage     | .gpkg     | 8%     | 地理空间数据存储和交换，基于SQLite，适用于开源GIS |
| MapInfo TAB    | .tab      | 5%     | 地理空间数据存储和交换，主要用于MapInfo软件     |
| VCT            | -         | 5%     | 国内土地项目专用矢量交换格式                |
| GML            | .gml      | 3%     | 地理空间数据交换，支持复杂地理对象             |
| MVT            | .mvt      | 3%     | 矢量瓦片格式，用于Web地图和移动应用           |
| GPX            | .gpx      | 1%     | GPS数据交换，如运动轨迹记录               |

### GDAL栅格数据支持（已去除没有地理参考的格式）
#### 常用栅格格式
| 格式名           | 扩展名        | 使用市场占比 | 主要用途                  |
| ------------- | ---------- | ------ | --------------------- |
| GeoTIFF       | .tif/.tiff | 35%    | 遥感影像、数字高程模型（DEM）、地形分析 |
| ERDAS Imagine | .img       | 15%    | 遥感影像处理、多波段数据存储        |
| JPEG2000      | .jp2       | 10%    | 高分辨率遥感影像存储，支持无损和有损压缩  |
| ASCII Grid    | .asc       | 10%    | 数字高程模型（DEM）、地形分析、土地覆盖 |
| ESRI Grid     | .grid      | 10%    | 地形分析、土地覆盖、气候研究        |
| NetCDF        | .nc        | 10%    | 气候和海洋科学数据存储           |
| HDF           | .hdf       | 5%     | 多维数据存储，适用于复杂数据结构      |
| PNG           | .png       | 5%     | 地图图像存储，无损压缩           |
| BMP           | .bmp       | 5%     | 简单地图图像处理，不包含地理定位信息    |

## 功能
### 地图展示
#### 小文件高精度展示
##### 矢量绘制
> 依赖QGIS

##### 栅格绘制
> 依赖QGIS

#### 大文件抽象展示
##### 矢量
> 地图抽象（难）。
> 1. 抽象策略的选择。
> 2. 一次性无法将整个文件读入，无法进行整体综合。

##### 栅格
> 图像下采样
> 1. 可能传统的图像处理算法库不适用超大图像文件。

### 矢量数据
#### 获取文件基本数据
1. 获取文件的描述信息
  1. shp：几何特征字段，位置
  2. geojson：名称字段，位置
  3. DEM模型：分辨率
  4. ...

#### 敏感点位
> 1. 输入目标点位Points/`*.shp/*.geojson/...`文件->ogr::Geometry对象，输入检测半径。
> 2. 输入待检查文件`*.shp/*.geojson/...`或目录。
> 3. 创建目标的缓冲区。
> 4. 创建待检查对象的ogr::Geometry对象，坐标投影转换到目标坐标系，进行缓冲区测试。
> 5. 返回所有测试结果。

#### 敏感对象
> 1. 输入目标文件`*.shp/*.geojson/...`或目录，输入检测半径。
> 2. 输入待检查文件`*.shp/*.geojson/...`或目录。
> 3. 创建目标的缓冲区。
> 4. 创建待检查对象的ogr::Geometry对象，坐标投影转换到坐标系，进行缓冲区测试。
> 5. 返回所有测试结果。

### 栅格数据
#### 分辨率
> 1. 输入图片，读取图片， Getgeotransform(double* adGeotransform)。
> 2. 区域范围：adGeotransform[0] （左上角的x坐标）,adGeotransform[3] (左上角的y坐标)。
> 3. 东西向分辨率：adGeotransform[1], 南北向分辨率adGeotransform[5] 单位：$m/pixel$。

#### 敏感区域
> 1. 输入图片，读取图像。
> 2. 输入区域范围的多边形Polygon。
> 3. 计算图像$Image \cap Polygon = \emptyset$。

#### 区域检测
> 1. 输入目标图片。
> 2. 输入待检查图片或者目录。
> 3. 检查每一张待检查图片中是否有与目标图片高度相似的部分。（图像匹配 SIFT,Moorvec,...）

### 功能依赖：
>1. gdal/ogr-数据格式转换/矢量分析。
>2. OpenCV-栅格数据计算。
>3. GEOS-矢量数据计算。

## 前端
### QGIS(Qt)

## 附录
### GDAL所有可支持的文件格式

| 简称             | 长名称                               | 创造      | 拷贝      | 地理参考    | 生成要求                                 |
| -------------- | --------------------------------- | ------- | ------- | ------- | ------------------------------------ |
| AAIGrid        | Arc/Info ASCII Grid               | 不       | **Yes** | **Yes** | 默认内置                                 |
| ACE2           | ACE2级                             | 不       | 不       | **Yes** | 默认内置                                 |
| ADRG           | ADRG/ARC数字化栅格图形（.gen/.thf）        | **Yes** | **Yes** | **Yes** | 默认内置                                 |
| AIG            | Arc/Info二进制网格                     | 不       | 不       | **Yes** | 默认内置                                 |
| AIRSAR         | AIRSAR极化格式                        | 不       | 不       | 不       | 默认内置                                 |
| ARG            | Azavea栅格                          | **Yes** | **Yes** | **Yes** | 默认内置                                 |
| BAG            | 水深属性网格                            | **Yes** | **Yes** | **Yes** | libhdf5文件                            |
| BLX            | 麦哲伦BLX拓扑文件格式                      | 不       | **Yes** | **Yes** | 默认内置                                 |
| BMP            | Microsoft Windows设备无关位图           | **Yes** | **Yes** | **Yes** | 默认内置                                 |
| BSB            | Maptech/NOAA BSB海图格式              | 不       | 不       | **Yes** | 默认内置                                 |
| BT             | bt二进制地形格式                         | **Yes** | **Yes** | **Yes** | 默认内置                                 |
| BYN            | 加拿大自然资源部的大地水准面文件格式（.byn）          | **Yes** | **Yes** | **Yes** | 默认内置                                 |
| CAD            | AutoCAD DWG栅格图层                   | 不       | 不       | **Yes** | （提供内部libopencad）                     |
| CALS           | CALS类型1                           | 不       | **Yes** | 不       | 默认内置                                 |
| CEOS           | CEO形象                             | 不       | **Yes** | 不       | 默认内置                                 |
| COASP          | DRDC-COASP SAR处理器栅格               | 不       | 不       | 不       | 默认内置                                 |
| COG            | 云优化GeoTIFF生成器                     | 不       | **Yes** | **Yes** | 默认内置                                 |
| COSAR          | TerraSAR-X复杂SAR数据产品               | 不       | 不       | 不       | 默认内置                                 |
| CPG            | Convair Polgaps数据                 | 不       | 不       | **Yes** | 默认内置                                 |
| CTable2        | CTable2基准网格偏移                     | **Yes** | **Yes** | **Yes** | 默认内置                                 |
| CTG            | USGS-LULC复合主题网格                   | 不       | 不       | **Yes** | 默认内置                                 |
| DAAS           | DAAS（空中客车DS智能数据服务驱动程序）            | 不       | 不       | **Yes** | 利勃曲尔                                 |
| DDS            | 直接绘制曲面                            | 不       | **Yes** | 不       | 紧缩库                                  |
| DERIVED        | 派生子数据集驱动程序                        | 不       | 不       | 不       | 默认内置                                 |
| DIMAP          | 斑点尺寸                              | 不       | 不       | **Yes** | 默认内置                                 |
| DIPEx          | 弹性双峰                              | 不       | 不       | **Yes** | 默认内置                                 |
| DODS           | OPeNDAP网格客户端                      | 不       | 不       | **Yes** | 利伯达普                                 |
| DOQ1           | 第一代USGS DOQ                       | 不       | 不       | **Yes** | 默认内置                                 |
| DOQ2           | 新标签USGS DOQ                       | 不       | 不       | **Yes** | 默认内置                                 |
| DTED           | 军事高程资料                            | 不       | **Yes** | **Yes** | 默认内置                                 |
| ECRGTOC        | ECRG目录（TOC.xml）                   | 不       | 不       | **Yes** | 默认内置                                 |
| ECW            | 增强压缩小波（.ecw）                      | **Yes** | **Yes** | **Yes** | ECW软件开发包                             |
| EEDAI          | 谷歌地球引擎数据API图片                     | 不       | 不       | **Yes** | 利勃曲尔                                 |
| EHdr           | ESRI.hdr Labelled                 | **Yes** | **Yes** | **Yes** | 默认内置                                 |
| EIR            | 埃尔达斯想象生                           | 不       | 不       | **Yes** | 默认内置                                 |
| ELAS           | 地球资源实验室应用软件                       | **Yes** | **Yes** | **Yes** | 默认内置                                 |
| ENVI           | ENVI. HDR Labelled Raster         | **Yes** | **Yes** | **Yes** | 默认内置                                 |
| ERS            | ERMapper.ERS公司                    | **Yes** | **Yes** | **Yes** | 默认内置                                 |
| ESAT           | Envisat图像产品                       | 不       | 不       | 不       | 默认内置                                 |
| ESRIC          | Esri压缩缓存                          | 不       | 不       | **Yes** | 默认内置                                 |
| EXR            | 扩展动态范围图像文件格式                      | **Yes** | **Yes** | **Yes** | libopenexr软件                         |
| FAST           | EOSAT快速格式                         | 不       | 不       | **Yes** | 默认内置                                 |
| FIT            | FIT                               | 不       | **Yes** | **Yes** | 默认内置                                 |
| FITS           | 灵活的图像传输系统                         | **Yes** | **Yes** | **Yes** | 利比西奥                                 |
| GenBin         | 通用二进制文件（.hdr标签）                   | 不       | 不       | 不       | 默认内置                                 |
| GeoRaster      | 甲骨文空间地理学家                         | **Yes** | **Yes** | **Yes** | Oracle客户端库                           |
| GFF            | Sandia国家实验室GSAT文件格式               | 不       | 不       | 不       | 默认内置                                 |
| GIF            | 图形交换格式                            | 不       | **Yes** | 不       | （提供内部GIF库）                           |
| GPKG           | 地质包栅格                             | **Yes** | **Yes** | **Yes** | libsqlite3（以及任何或所有PNG、JPEG、WEBP驱动程序） |
| GRASS          | GRASS 栅格格式                        | 不       | 不       | **Yes** | libgrass                             |
| GRASSASCIIGrid | 草ASCII网格                          | 不       | 不       | **Yes** | 默认内置                                 |
| GRIB           | WMO一般规则分布的二进制信息                   | 不       | 不       | **Yes** | 默认内置                                 |
| GS7BG          | Golden Software Surfer 7二进制网格文件格式 | **Yes** | **Yes** | **Yes** | 默认内置                                 |
| GSAG           | 黄金软件ASCII网格文件格式                   | 不       | 不       | **Yes** | 默认内置                                 |
| GSBG           | Golden软件二进制网格文件格式                 | **Yes** | **Yes** | **Yes** | 默认内置                                 |
| GSC            | GSC土工格栅                           | 不       | 不       | 不       | 默认内置                                 |
| GTA            | 通用标记数组                            | 不       | **Yes** | **Yes** | 利布塔                                  |
| GTiff          | GeoTIFF文件格式                       | **Yes** | **Yes** | **Yes** | 默认内置                                 |
| GXF            | 网格交换文件                            | 不       | 不       | **Yes** | 默认内置                                 |
| HDF4           | 分层数据格式版本4（HDF4）                   | **Yes** | **Yes** | **Yes** | 利比亚国防军                               |
| HDF5           | 分层数据格式版本5（HDF5）                   | 不       | 不       | **Yes** | libhdf5文件                            |
| HEIF           | ISO/IEC 23008-12:2017高效图像文件格式     | 不       | 不       | 不       | libheif（>=1.1），基于libde265构建          |
| HF2            | HF2/HFZ高场栅格                       | 不       | **Yes** | **Yes** | 默认内置                                 |
| HFA            | 艾达斯想象                             | **Yes** | **Yes** | **Yes** | 默认内置                                 |
| RST            | Idrisi栅格格式                        | **Yes** | **Yes** | **Yes** | 默认内置                                 |
| ILWIS          | 栅格地图                              | **Yes** | **Yes** | **Yes** | 默认内置                                 |
| IRIS           | 维萨拉气象雷达软件格式                       | 不       | 不       | **Yes** | 默认内置                                 |
| ISCE           | ISCE                              | **Yes** | **Yes** | **Yes** | 默认内置                                 |
| ISG            | 大地水准面国际服务                         | 不       | 不       | **Yes** | 默认内置                                 |
| ISIS2          | 美国地质勘探局天体地质ISIS立方体（第2版）           | **Yes** | **Yes** | **Yes** | 默认内置                                 |
| ISIS3          | 美国地质勘探局天体地质ISIS立方体（第3版）           | **Yes** | **Yes** | **Yes** | 默认内置                                 |
| JDEM           | 日本DEM（.mem）                       | 不       | 不       | **Yes** | 默认内置                                 |
| JP2ECW         | ERDAS JPEG2000（.jp2）              | **Yes** | **Yes** | **Yes** | ECW软件开发包                             |
| JP2KAK         | JPEG-2000（基于Kakadu）               | 不       | **Yes** | **Yes** |                                      |

### OGR可支持的全部矢量格式

| 简称             | 长名称                                  | 创造      | 地理参考    | 生成要求                                              |
| -------------- | ------------------------------------ | ------- | ------- | ------------------------------------------------- |
| AmigoCloud     | AmigoCloud                           | **Yes** | **Yes** | 利勃曲尔                                              |
| Arrow          | (Geo)Arrow IPC File Format / Stream  | **Yes** | **Yes** | Apache Arrow C++ library                          |
| AVCBIN         | Arc/Info Binary Coverage             | 不       | **Yes** | 默认内置                                              |
| AVCE00         | Arc/Info E00（ASCII）覆盖范围              | 不       | **Yes** | 默认内置                                              |
| CAD            | AutoCAD图纸                            | 不       | **Yes** | （提供内部libopencad）                                  |
| CARTO          | Carto                                | **Yes** | **Yes** | 利勃曲尔                                              |
| CSV            | Comma Separated Value (.csv)         | **Yes** | **Yes** | 默认内置                                              |
| CSW            | OGC CSW（Web目录服务）                     | 不       | **Yes** | 利勃曲尔                                              |
| DGN            | 微站DGN                                | **Yes** | **Yes** | 默认内置                                              |
| DGNv8          | 微型工作站DGN v8                          | **Yes** | **Yes** | 开放设计联盟泰加类库                                        |
| DODS           | DODS/OPeNDAP                         | 不       | **Yes** | 利伯达普                                              |
| DWG            | AutoCAD图纸                            | 不       | 不       | 开放设计联盟泰加类库                                        |
| DXF            | AutoCAD DXF                          | **Yes** | 不       | 默认内置                                              |
| EDIGEO         | EDIGEO                               | 不       | **Yes** | 默认内置                                              |
| EEDA           | 谷歌地球引擎数据API                          | 不       | **Yes** | 利勃曲尔                                              |
| Elasticsearch  | Elasticsearch：用于Elasticsearch的地理编码对象 | **Yes** | **Yes** | 利勃曲尔                                              |
| ESRIJSON       | ESRIJSON/FeatureService驱动程序          | 不       | **Yes** | 默认内置                                              |
| FileGDB        | ESRI文件地理数据库（FileGDB）                 | **Yes** | **Yes** | FileGDB API库                                      |
| FlatGeobuf     | FlatGeobuf                           | **Yes** | **Yes** | 默认内置                                              |
| Geoconcept     | GeoConcept文本导出                       | **Yes** | **Yes** | 默认内置                                              |
| GeoJSON        | GeoJSON                              | **Yes** | **Yes** | 默认内置                                              |
| GeoJSONSeq     | GeoJSONSeq：GeoJSON特性的序列              | **Yes** | **Yes** | 默认内置                                              |
| GeoRSS         | GeoRSS：RSS提要的地理编码对象                  | **Yes** | **Yes** | （读取支持需要libexpat）                                  |
| GML            | 地理标记语言                               | **Yes** | **Yes** | （读取支持需要Xerces或libexpat）                           |
| GMLAS          | 应用程序模式驱动的地理标记语言（GML）                 | 不       | **Yes** | 干燥                                                |
| GMT            | GMT ASCII矢量（.GMT）                    | **Yes** | **Yes** | 默认内置                                              |
| GPKG           | 地质包矢量                                | **Yes** | **Yes** | libsqlite3                                        |
| GPSBabel       | GPSBabel                             | **Yes** | **Yes** | （读取支持需要GPX驱动程序和libexpat）                          |
| GPX            | GPS交换格式                              | **Yes** | **Yes** | （读取支持需要libexpat）                                  |
| GRASS          | GRASS 矢量格式                           | 不       | **Yes** | libgrass                                          |
| HANA           | SAP HANA                             | **Yes** | **Yes** | odbc-cpp-wrapper                                  |
| IDB            | IDB                                  | **Yes** | **Yes** | Informix数据库                                       |
| IDRISI         | Idrisi Vector (.VCT)                 | 不       | **Yes** | 默认内置                                              |
| INTERLIS 1     | “INTERLIS 1”和“INTERLIS 2”驱动程序        | **Yes** | **Yes** | 干燥                                                |
| INTERLIS 2     | “INTERLIS 1”和“INTERLIS 2”驱动程序        | **Yes** | **Yes** | 干燥                                                |
| JML            | JML:OpenJUMP JML格式                   | **Yes** | **Yes** | （读取支持需要libexpat）                                  |
| KML            | 锁眼标记语言                               | **Yes** | **Yes** | （读取支持需要libexpat）                                  |
| LIBKML         | LIBKML驱动程序（.kml.kmz）                 | **Yes** | **Yes** | libkml语言                                          |
| LVBAG          | 荷兰卡达斯特LV包2.0提取物                      | 不       | 不       | libexpat公司                                        |
| MapML          | 地图管理语言                               | **Yes** | **Yes** | 默认内置                                              |
| Memory         | Memory                               | **Yes** | **Yes** | 默认内置                                              |
| MITAB          | MapInfo TAB和MIF/MID                  | **Yes** | **Yes** | 默认内置                                              |
| MongoDBv3      | MongoDBv3                            | **Yes** | **Yes** | Mongo CXX>=3.4.0客户端库                              |
| MSSQLSpatial   | Microsoft SQL Server空间数据库            | **Yes** | **Yes** | ODBC库                                             |
| MVT            | MVT:地图框矢量平铺                          | **Yes** | **Yes** | （需要SQLite和GEOS提供写支持）                              |
| MySQL          | MySQL                                | **Yes** | **Yes** | MySQL库                                            |
| NAS            | ALKIS                                | 不       | **Yes** | 干燥                                                |
| netCDF         | 矢量                                   | **Yes** | **Yes** | 伦敦银行同业拆借利率                                        |
| NGW            | 下一个网站                                | 不       | **Yes** | 利勃曲尔                                              |
| UK .NTF        | 英国NTF                                | 不       | **Yes** | 默认内置                                              |
| OAPIF          | OGC API-特性                           | 不       | **Yes** | 利勃曲尔                                              |
| OCI            | 空间数据库                                | **Yes** | **Yes** | 内控类库                                              |
| ODBC           | ODBC关系数据库                            | 不       | **Yes** | ODBC库                                             |
| ODS            | 打开文档电子表格                             | **Yes** | 不       | libexpat公司                                        |
| OGDI           | OGDI向量                               | 不       | **Yes** | 奥格迪类库                                             |
| OpenFileGDB    | ESRI文件地理数据库（OpenFileGDB）             | 不       | **Yes** | 默认内置                                              |
| OSM            | OpenStreetMap XML和PBF                | 不       | **Yes** | libsqlite3（和libexpat for OSM XML）                 |
| Parquet        | (Geo)Parquet                         | **Yes** | **Yes** | Parquet component of the Apache Arrow C++ library |
| PDF            | 地理空间PDF                              | **Yes** | **Yes** | 无用于写支持，Poppler/PoDoFo/PDFium用于读支持                 |
| PDS            | 行星数据系统表                              | 不       | 不       | 默认内置                                              |
| PostgreSQL     | PostgreSQL/邮政地理信息系统                  | **Yes** | **Yes** | PostgreSQL客户端库（libpq）                             |
| PGDump         | PostgreSQL SQL转储                     | **Yes** | **Yes** | 默认内置                                              |
| PGeo           | ESRI个人地理数据库                          | 不       | **Yes** | ODBC库                                             |
| PLScenes       | PLScenes（行星实验室场景/目录API）              | 不       | 不       | 利勃曲尔                                              |
| S57            | IHO S-57（附件）                         | 不       | **Yes** | 默认内置                                              |
| SDTS           | SDTS                                 | 不       | **Yes** | 默认内置                                              |
| Selafin        | Selafin files                        | **Yes** | **Yes** | 默认内置                                              |
| ESRI Shapefile | ESRI形状文件/DBF                         | **Yes** | **Yes** | 默认内置                                              |
| SOSI           | 挪威SOSI标准                             | 不       | 不       | FYBA类库                                            |
| SQLite         | SQLite/spacealite关系数据库管理系统           | **Yes** | **Yes** | libsqlite3或libspacealite                          |
| SVG            | 可缩放矢量图形                              | 不       | **Yes** | libexpat公司                                        |
| SXF            | SXF                                  | 不       | **Yes** | 默认内置                                              |
| TIGER          | U、 美国人口普查虎/线                         | 不       | **Yes** | 默认内置                                              |
| TopoJSON       | TopoJSON驱动程序                         | 不       | **Yes** | 默认内置                                              |
| VDV            | VDV-451/VDV-452/INTREST数据格式          | **Yes** | **Yes** | 默认内置                                              |
| VFK            | 捷克地籍交换数据格式                           | 不       | **Yes** | libsqlite3                                        |
| VRT            | 虚拟格式                                 | **Yes** | **Yes** | 默认内置                                              |
| WAsP           | WAsP.map格式                           | **Yes** | **Yes** | 默认内置                                              |
| WFS            | OGC WFS服务                            | 不       | **Yes** | 利勃曲尔                                              |
| XLS            | 名表                                   | 不       | 不       | 伦敦银行同业拆借利率                                        |
| XLSX           | MS Office开放XML电子表格                   |         |         |                                                   |
