package geography.agents;

import java.awt.image.RenderedImage;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import javax.media.jai.iterator.RandomIter;
import javax.media.jai.iterator.RandomIterFactory;
import org.geotools.coverage.grid.GridCoverage2D;
import org.locationtech.jts.geom.*;
import org.locationtech.jts.geom.prep.PreparedGeometry;
import org.locationtech.jts.geom.prep.PreparedGeometryFactory;
import repast.simphony.context.Context;
import repast.simphony.space.gis.Geography;

public class ModelManager {

    private static ModelManager instance;
    private Geometry drainageBasin;
    private Geometry rangelandAreas;
    private PreparedGeometry preparedBasin;
    private PreparedGeometry preparedRangeland;
    private GeometryFactory geomFactory = new GeometryFactory();

    private DirectRaster rangelandsRaster;
    private DirectRaster slopeRaster;
    private DirectRaster waterRaster;

    private GridCoverage2D[] rawRainfallRasters;
    private GridCoverage2D[] rawSaviRasters;

    private DirectRaster currentRainfallRaster;
    private DirectRaster currentSaviRaster;
    private int currentMonthLoaded = -1;

    private Context<Object> context;
    private Geography<Object> geography;
    private Map<String, List<Coordinate>> visitedLocations = new HashMap<>();
    private Map<String, Integer> exitedAgents = new HashMap<>();

    // 5‑meter spatial hash for fast proximity checks
    private final double CELL_SIZE = 5.0;
    private Map<Long, List<Coordinate>> spatialIndex = new ConcurrentHashMap<>();

    public static final double NODATA_VALUE = -9999.0;

    private ModelManager() {}

    public static ModelManager getInstance() {
        if (instance == null)
            instance = new ModelManager();
        return instance;
    }

    public void initialize(
            Geometry drainageBasin, Geometry rangelandAreas,
            GridCoverage2D rangelandsRaster, GridCoverage2D[] rainfallRasters,
            GridCoverage2D[] saviRasters, GridCoverage2D slopeRaster,
            GridCoverage2D waterRaster, Context<Object> context, Geography<Object> geography
    ) {
        this.drainageBasin = drainageBasin;
        this.rangelandAreas = rangelandAreas;
        if (drainageBasin != null) this.preparedBasin = PreparedGeometryFactory.prepare(drainageBasin);
        if (rangelandAreas != null) this.preparedRangeland = PreparedGeometryFactory.prepare(rangelandAreas);

        this.rawRainfallRasters = rainfallRasters;
        this.rawSaviRasters = saviRasters;
        this.context = context;
        this.geography = geography;

        this.visitedLocations.clear();
        this.spatialIndex.clear();
        this.exitedAgents.clear();
        this.currentMonthLoaded = -1;

        System.out.println("Initializing Cached Raster Engine...");
        this.rangelandsRaster = new DirectRaster(rangelandsRaster);
        this.slopeRaster = new DirectRaster(slopeRaster);
        this.waterRaster = new DirectRaster(waterRaster);

        checkAndLoadMonthData(0);
    }

    public DirectRaster getRangelandsRaster() { return rangelandsRaster; }
    public DirectRaster getSlopeRaster() { return slopeRaster; }
    public DirectRaster getWaterRaster() { return waterRaster; }
    public DirectRaster getRainfallRaster(int month) { checkAndLoadMonthData(month); return currentRainfallRaster; }
    public DirectRaster getSaviRaster(int month) { checkAndLoadMonthData(month); return currentSaviRaster; }

    private void checkAndLoadMonthData(int month) {
        if (month < 0 || month >= rawRainfallRasters.length) return;
        if (month != currentMonthLoaded) {
            System.out.println("Switching to Month Index: " + month);
            System.gc();
            this.currentRainfallRaster = new DirectRaster(rawRainfallRasters[month]);
            this.currentSaviRaster = new DirectRaster(rawSaviRasters[month]);
            this.currentMonthLoaded = month;
        }
    }

    // ---- Spatial hash methods for 5‑m proximity ----
    private long getGridKey(double x, double y) {
        long col = (long) Math.floor(x / CELL_SIZE);
        long row = (long) Math.floor(y / CELL_SIZE);
        return (col << 32) | (row & 0xffffffffL);
    }

    public boolean isTooCloseToOthers(double x, double y, double minDist) {
        long col = (long) Math.floor(x / CELL_SIZE);
        long row = (long) Math.floor(y / CELL_SIZE);
        double minDistSq = minDist * minDist;

        for (long c = col - 1; c <= col + 1; c++) {
            for (long r = row - 1; r <= row + 1; r++) {
                long key = (c << 32) | (r & 0xffffffffL);
                List<Coordinate> bucket = spatialIndex.get(key);
                if (bucket != null) {
                    for (Coordinate coord : bucket) {
                        double dx = coord.x - x;
                        double dy = coord.y - y;
                        if ((dx * dx + dy * dy) < minDistSq) {
                            return true;
                        }
                    }
                }
            }
        }
        return false;
    }

    public synchronized void addOccupiedCoordinate(Coordinate coord) {
        long key = getGridKey(coord.x, coord.y);
        spatialIndex.computeIfAbsent(key, k -> new ArrayList<>()).add(coord);
    }

    public synchronized void removeOccupiedCoordinate(Coordinate coord) {
        long key = getGridKey(coord.x, coord.y);
        List<Coordinate> bucket = spatialIndex.get(key);
        if (bucket != null) {
            bucket.remove(coord);
            if (bucket.isEmpty()) spatialIndex.remove(key);
        }
    }

    // 🚨 V7 RULE: Proximity-weighted water access (W_proximity_i)
    public double getMaxWaterInRadius(double centerX, double centerY, double radiusMeters) {
        if (waterRaster == null) return 0.0;

        double maxWater = 0.0;
        double step = 1000.0; // Sample every 1km for high-speed performance

        for (double dx = -radiusMeters; dx <= radiusMeters; dx += step) {
            for (double dy = -radiusMeters; dy <= radiusMeters; dy += step) {
                if ((dx * dx + dy * dy) <= (radiusMeters * radiusMeters)) {
                    double val = waterRaster.getValue(centerX + dx, centerY + dy);
                    if (val != NODATA_VALUE && val > maxWater) {
                        maxWater = val;
                    }
                }
            }
        }
        return maxWater;
    }

    // ---- Legacy proximity methods (still used for initial placement) ----
    public boolean isValidLocation(double x, double y) {
        double rangelandValue = rangelandsRaster.getValue(x, y);
        if (Math.abs(rangelandValue - 1.0) > 0.001) return false;

        double slopeValue = slopeRaster.getValue(x, y);
        if (Math.abs(slopeValue - 1.0) > 0.001) return false;

        return isNonSedentary(x, y);
    }

    public boolean isValidLocation(Point point) {
        if (point == null) return false;
        return isValidLocation(point.getX(), point.getY());
    }

    public boolean isNonSedentary(double x, double y) {
        if (preparedRangeland == null) return true;
        return preparedRangeland.contains(geomFactory.createPoint(new Coordinate(x, y)));
    }

    public void markVisited(Point point, String agentId) {
        if (point == null || agentId == null) return;
        visitedLocations.computeIfAbsent(agentId, k -> new ArrayList<>()).add(point.getCoordinate());
    }

    public boolean isVisited(Coordinate coord, String agentId) {
        List<Coordinate> visited = visitedLocations.getOrDefault(agentId, Collections.emptyList());
        for (Coordinate c : visited) {
            if (c.distance(coord) < 1.0) return true;
        }
        return false;
    }

    public void trackExit(NomadicHousehold agent, int month) {
        if (agent == null) return;
        Point point = (Point) geography.getGeometry(agent);
        boolean outsideBasin = (point != null && preparedBasin != null && !preparedBasin.contains(point));
        if (point == null || outsideBasin) {
            exitedAgents.putIfAbsent(agent.getId(), month);
        }
    }

    public Integer getAgentExitMonth(String agentId) {
        return exitedAgents.get(agentId);
    }

    public double getRasterValueAtCoord(double x, double y, DirectRaster raster) {
        if (raster == null) return NODATA_VALUE;
        return raster.getValue(x, y);
    }

    public Geography<Object> getGeography() { return geography; }

    // ---- DirectRaster inner class (unchanged) ----
    public static class DirectRaster {
        private final RandomIter iter;
        private final int width, height;
        private final double minX, maxY, pixelWidth, pixelHeight;
        private final Map<Long, Double> pixelCache;
        private final int MAX_CACHE = 100000;

        public DirectRaster(GridCoverage2D coverage) {
            if (coverage == null) {
                this.iter = null; this.width = 0; this.height = 0; this.minX = 0; this.maxY = 0; this.pixelWidth = 0; this.pixelHeight = 0;
                this.pixelCache = null;
                return;
            }
            RenderedImage img = coverage.getRenderedImage();
            this.iter = RandomIterFactory.create(img, null);
            this.width = img.getWidth();
            this.height = img.getHeight();
            org.geotools.geometry.Envelope2D env = coverage.getEnvelope2D();
            this.minX = env.getMinX();
            this.maxY = env.getMaxY();
            this.pixelWidth = env.getWidth() / (double) width;
            this.pixelHeight = env.getHeight() / (double) height;

            this.pixelCache = Collections.synchronizedMap(
                new LinkedHashMap<Long, Double>(MAX_CACHE + 1, 1.0f, true) {
                    @Override
                    protected boolean removeEldestEntry(Map.Entry<Long, Double> eldest) {
                        return size() > MAX_CACHE;
                    }
                }
            );
        }

        public double getValue(double x, double y) {
            int col = (int) ((x - minX) / pixelWidth);
            int row = (int) ((maxY - y) / pixelHeight);
            if (col < 0 || col >= width || row < 0 || row >= height) return NODATA_VALUE;
            long key = (((long) col) << 32) | (row & 0xffffffffL);
            Double cached = pixelCache.get(key);
            if (cached != null) return cached;
            try {
                double val = iter.getSampleDouble(col, row, 0);
                if (val == -9999.0 || Double.isNaN(val) || val < -9000 || val > 100000.0) val = NODATA_VALUE;
                pixelCache.put(key, val);
                return val;
            } catch (Exception e) {
                return NODATA_VALUE;
            }
        }
    }
}