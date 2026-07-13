package geography.agents;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.net.URL;
import org.geotools.coverage.grid.GridCoverage2D;
import org.geotools.coverage.grid.io.AbstractGridCoverage2DReader;
import org.geotools.coverage.grid.io.AbstractGridFormat;
import org.geotools.coverage.grid.io.GridFormatFinder;
import org.geotools.data.shapefile.ShapefileDataStore;
import org.geotools.data.simple.SimpleFeatureIterator;
import org.opengis.feature.simple.SimpleFeature;
import org.locationtech.jts.geom.*;
import org.locationtech.jts.geom.prep.PreparedGeometry;
import org.locationtech.jts.geom.prep.PreparedGeometryFactory;
import repast.simphony.context.Context;
import repast.simphony.context.space.gis.GeographyFactoryFinder;
import repast.simphony.dataLoader.ContextBuilder;
import repast.simphony.engine.environment.RunEnvironment;
import repast.simphony.engine.schedule.IAction;
import repast.simphony.engine.schedule.ISchedule;
import repast.simphony.engine.schedule.ScheduleParameters;
import repast.simphony.parameter.Parameters;
import repast.simphony.random.RandomHelper;
import repast.simphony.space.gis.Geography;
import repast.simphony.space.gis.GeographyParameters;

public class ContextCreator implements ContextBuilder<Object> {

    private static Geometry STATIC_drainageBasin;
    private static Geometry STATIC_rangelandAreas;
    
    private static GridCoverage2D STATIC_rangelandsRaster;
    private static GridCoverage2D[] STATIC_rainfallRasters = new GridCoverage2D[12];
    private static GridCoverage2D[] STATIC_saviRasters = new GridCoverage2D[12];
    
    private static GridCoverage2D STATIC_slopeRaster;
    private static GridCoverage2D STATIC_waterRaster;

    private GeometryFactory geomFactory = new GeometryFactory();
    
    private String[] months = {
        "jan", "feb", "mar", "apr", "may", "jun", 
        "jul", "aug", "sep", "oct", "nov", "dec"
    };

    @Override
    public Context<Object> build(Context<Object> context) {
        
        Parameters parm = RunEnvironment.getInstance().getParameters();
        String movementStrategy = parm.getString("movementStrategy");
        
        int seed = 1; 
        if (parm.getSchema().contains("randomSeed")) {
            seed = (Integer) parm.getValue("randomSeed"); 
        }
        
        String dataYear = "20"; 
        if (parm.getSchema().contains("dataYear")) {
            dataYear = parm.getString("dataYear");
        }
        
        System.out.println("=== STARTING SIMULATION (V6: DYNAMIC ALLOCATION) ===");
        System.out.println("Strategy: " + movementStrategy);

        GeographyParameters<Object> geoParams = new GeographyParameters<>();
        Geography<Object> geography = GeographyFactoryFinder.createGeographyFactory(null)
                .createGeography("Geography", context, geoParams);
        
        if (STATIC_drainageBasin == null) {
            System.out.println("Loading GIS layers from disk...");
            loadAvailableGISLayers(context, geography, dataYear);
        } else {
            System.out.println("Using cached GIS layers...");
            reAddStaticAgents(context, geography);
        }
        
        if (STATIC_drainageBasin == null) {
            System.err.println("FATAL ERROR: basin.shp failed to load.");
            RunEnvironment.getInstance().endRun();
            return context;
        }
        
        ModelManager manager = ModelManager.getInstance();
        manager.initialize(STATIC_drainageBasin, STATIC_rangelandAreas, STATIC_rangelandsRaster, STATIC_rainfallRasters, 
                             STATIC_saviRasters, STATIC_slopeRaster, STATIC_waterRaster, 
                             context, geography);
        
        context.add(manager);
        RandomHelper.setSeed(seed);
        
        System.out.println("Preparing Global Spatial Indexes...");
        PreparedGeometry prepBasin = PreparedGeometryFactory.prepare(STATIC_drainageBasin);
        PreparedGeometry prepRange = null;
        if (STATIC_rangelandAreas != null) {
            prepRange = PreparedGeometryFactory.prepare(STATIC_rangelandAreas);
        }

        System.out.println("Loading districts_pop.shp for dynamic agent placement...");
        int totalAgentsCreated = 0;
        
        try {
            URL districtUrl = new File("C:/RepastData/districts_pop.shp").toURL();
            ShapefileDataStore districtStore = new ShapefileDataStore(districtUrl);
            SimpleFeatureIterator districtIter = districtStore.getFeatureSource().getFeatures().features();
            
            while (districtIter.hasNext()) {
                SimpleFeature feature = districtIter.next();
                Geometry districtGeom = (Geometry) feature.getDefaultGeometry();
                
                Object agentNumbObj = feature.getAttribute("agent_numb");
                if (agentNumbObj == null) agentNumbObj = feature.getAttribute("agent_num");

                int targetAgents = 0;
                if (agentNumbObj instanceof Number) {
                    targetAgents = ((Number) agentNumbObj).intValue();
                } else if (agentNumbObj instanceof String) {
                    targetAgents = Integer.parseInt(((String) agentNumbObj).trim());
                }
                
                if (targetAgents <= 0) continue;
                
                System.out.println("Allocating " + targetAgents + " agents to district feature...");
                
                PreparedGeometry prepDistrict = PreparedGeometryFactory.prepare(districtGeom);
                Envelope env = districtGeom.getEnvelopeInternal();
                double minX = env.getMinX();
                double maxX = env.getMaxX();
                double minY = env.getMinY();
                double maxY = env.getMaxY();

                int agentsInDistrict = 0;
                // Increased attempts from 10000 to 100000 to improve placement success
                int maxAttempts = targetAgents * 100000;
                int attempts = 0;

                while (agentsInDistrict < targetAgents && attempts < maxAttempts) {
                    attempts++;
                    
                    double randomX = minX + (RandomHelper.nextDouble() * (maxX - minX));
                    double randomY = minY + (RandomHelper.nextDouble() * (maxY - minY));
                    Point pt = geomFactory.createPoint(new Coordinate(randomX, randomY));
                    
                    if (!prepDistrict.contains(pt)) continue;
                    if (!prepBasin.contains(pt)) continue;
                    if (prepRange != null && !prepRange.contains(pt)) continue;
                    if (!manager.isValidLocation(randomX, randomY)) continue;
                    // Original design: 5 metre minimum distance between agents
                    if (manager.isTooCloseToOthers(randomX, randomY, 5.0)) continue;
                    
                    NomadicHousehold agent = new NomadicHousehold("Household_" + totalAgentsCreated, geography, manager);
                    
                    context.add(agent);
                    geography.move(agent, pt);
                    
                    agent.initializeLocation();
                    manager.markVisited(pt, agent.getId());
                    
                    agentsInDistrict++;
                    totalAgentsCreated++;
                }
                
                if (agentsInDistrict < targetAgents) {
                    System.err.println("WARNING: Could only fit " + agentsInDistrict + " out of " + targetAgents + " in this district due to space constraints.");
                }
            }
            
            districtIter.close();
            districtStore.dispose();
            
        } catch (Exception e) {
            System.err.println("FATAL ERROR: Failed to load or process districts_pop.shp");
            e.printStackTrace();
            RunEnvironment.getInstance().endRun();
            return context;
        }

        System.out.println("V6 Agent initialization complete. Total active agents: " + totalAgentsCreated);

        ISchedule schedule = RunEnvironment.getInstance().getCurrentSchedule();
        
        for (int tick = 1; tick <= 12; tick++) {
            ScheduleParameters sp = ScheduleParameters.createOneTime(tick, ScheduleParameters.LAST_PRIORITY);
            final int t = tick;
            schedule.schedule(sp, new IAction() {
                @Override
                public void execute() {
                    writeMonthlyAgentLocations(context, t);
                }
            });
        }

        ScheduleParameters endParams = ScheduleParameters.createOneTime(12.0, ScheduleParameters.LAST_PRIORITY);
        schedule.schedule(endParams, new IAction() {
            @Override
            public void execute() {
                exportMetrics(context);
                System.out.println(">>> Automated Pipeline Complete. Handing control back to Python... <<<");
                RunEnvironment.getInstance().endRun(); 
            }
        });

        return context;
    }

    private synchronized void writeMonthlyAgentLocations(Context<Object> context, int tick) {
        String filename = "v5_custom_locations.csv"; 
        boolean fileExists = new File(filename).exists();
        try (PrintWriter writer = new PrintWriter(new FileWriter(filename, true))) {
            if (!fileExists) {
                writer.println("run,Id,X,Y,Moving,CurrentRainfall,CurrentSavi,MovementStrategy,month,ExitMonth");
            }
            for (Object obj : context) {
                if (obj instanceof NomadicHousehold) {
                    NomadicHousehold agent = (NomadicHousehold) obj;
                    writer.printf(java.util.Locale.US, "1,%s,%.2f,%.2f,%b,%.6f,%.6f,%s,%d,%d\n",
                        agent.getId(),
                        agent.getX(),
                        agent.getY(),
                        agent.isMoving(),
                        agent.getCurrentRainfall(),
                        agent.getCurrentSavi(),
                        agent.getMovementStrategy(),
                        tick,
                        agent.getExitMonth()
                    );
                }
            }
        } catch (Exception e) {
            System.err.println("Failed to write monthly locations CSV!");
            e.printStackTrace();
        }
    }

    private void exportMetrics(Context<Object> context) {
        System.out.println("Exporting results to output_metrics.csv...");
        try (PrintWriter writer = new PrintWriter(new File("output_metrics.csv"))) {
            writer.println("AgentID,DistanceMovedKm,ExitMonth,FinalX,FinalY");
            
            for (Object obj : context) {
                if (obj instanceof NomadicHousehold) {
                    NomadicHousehold agent = (NomadicHousehold) obj;
                    writer.printf(java.util.Locale.US, "%s,%.2f,%d,%.2f,%.2f\n", 
                        agent.getId(), 
                        agent.getDistanceMovedKm(), 
                        agent.getExitMonth(), 
                        agent.getX(), 
                        agent.getY());
                }
            }
            System.out.println(">>> output_metrics.csv successfully written! <<<");
        } catch (Exception e) {
            System.err.println("Failed to write CSV output!");
            e.printStackTrace();
        }
    }

    private void reAddStaticAgents(Context<Object> context, Geography<Object> geography) {
        if (STATIC_drainageBasin != null) {
            Basin basin = new Basin(STATIC_drainageBasin);
            context.add(basin);
            geography.move(basin, STATIC_drainageBasin);
        }
        if (STATIC_rangelandAreas != null) {
            Rangeland rangeland = new Rangeland(STATIC_rangelandAreas);
            context.add(rangeland);
            geography.move(rangeland, STATIC_rangelandAreas);
        }
        if (STATIC_rangelandsRaster != null) geography.addCoverage("rangelands", STATIC_rangelandsRaster);
        if (STATIC_slopeRaster != null) geography.addCoverage("slope", STATIC_slopeRaster);
        if (STATIC_waterRaster != null) geography.addCoverage("water_sources", STATIC_waterRaster);
    }
    
    private void loadAvailableGISLayers(Context<Object> context, Geography<Object> geography, String dataYear) {
        String basePath = "C:/RepastData/";

        STATIC_drainageBasin = loadShapefileSingle(basePath + "basin.shp");
        context.add(new Basin(STATIC_drainageBasin));
        geography.move(new Basin(STATIC_drainageBasin), STATIC_drainageBasin);

        STATIC_rangelandAreas = loadShapefileSingle(basePath + "nonsedentaryareas_utm.shp");
        context.add(new Rangeland(STATIC_rangelandAreas));
        geography.move(new Rangeland(STATIC_rangelandAreas), STATIC_rangelandAreas);

        STATIC_rangelandsRaster = loadRasterLayer(basePath + "rangelands_resampled_utm.tif");
        geography.addCoverage("rangelands", STATIC_rangelandsRaster);

        STATIC_slopeRaster = loadRasterLayer(basePath + "slopepercent_reclass_utm.tif");
        geography.addCoverage("slope", STATIC_slopeRaster);

        STATIC_waterRaster = loadRasterLayer(basePath + "water_sources_2020_utm.tif");
        geography.addCoverage("water_sources", STATIC_waterRaster);

        String saviYearPrefix = "20" + dataYear; 
        
        for (int i = 0; i < 12; i++) {
            String month = months[i];
            
            String rainPath = basePath + "rainfall_" + month + dataYear + "resampled_utm.tif";
            String saviPath = basePath + "savi_" + month + saviYearPrefix + "notnorm_utm.tif";
            
            STATIC_rainfallRasters[i] = loadRasterLayer(rainPath);
            if (STATIC_rainfallRasters[i] == null) System.err.println("CRITICAL WARNING: Missing " + rainPath);

            STATIC_saviRasters[i] = loadRasterLayer(saviPath);
             if (STATIC_saviRasters[i] == null) System.err.println("CRITICAL WARNING: Missing " + saviPath);
        }
    }

    private Geometry loadShapefileSingle(String filename) {
        try {
            URL url = new File(filename).toURL();
            ShapefileDataStore store = new ShapefileDataStore(url);
            SimpleFeatureIterator fiter = store.getFeatureSource().getFeatures().features();
            Geometry geom = null;
            if (fiter.hasNext()) geom = (Geometry) fiter.next().getDefaultGeometry();
            fiter.close();
            store.dispose();
            return geom;
        } catch (Exception e) { return null; }
    }

    private GridCoverage2D loadRasterLayer(String filename) {
        try {
            File file = new File(filename);
            AbstractGridFormat format = GridFormatFinder.findFormat(file);
            if (format == null) return null;
            AbstractGridCoverage2DReader reader = format.getReader(file);
            return reader.read(null);
        } catch (Exception e) { return null; }
    }
}